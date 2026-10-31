"""Which node stores a key, and what AddNode / pin do to that map.

In the readings: partitioning / sharding — Think Distributed Systems ch. 7;
Designing Data-Intensive Applications ch. 7.

See week06/README.md. Useful imports: ``from platform import node_pb2``
and ``from platform.util import mode, RpcFailed``.
"""
from __future__ import annotations
import hashlib
import threading
from platform import node_pb2
from platform.node import Node
from platform.util import RpcFailed, mode
from .base import PluginBase

def _hash(key: bytes) -> int:
    # Not hash(): Python randomizes it per process, so nodes would disagree.
    return int.from_bytes(hashlib.md5(key).digest(), "big")

def _slot_table(join_order: tuple[str, ...], n_slots: int) -> list[str]:
    # Each node that joins takes n_slots // (members so far) slots from
    # whoever holds the most, so a new node only moves its own share.
    table = [join_order[0]] * n_slots
    counts = {join_order[0]: n_slots}
    for k, newcomer in enumerate(join_order[1:], start=2):
        counts[newcomer] = 0
        earlier = join_order[: k - 1]
        for _ in range(n_slots // k):
            donor = max(earlier, key=lambda n: (counts[n], -earlier.index(n)))
            slot = max(i for i in range(n_slots) if table[i] == donor)
            table[slot] = newcomer
            counts[donor] -= 1
            counts[newcomer] += 1
    return table

class Plugin(PluginBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._join_order: list[str] | None = None
        self._table_for: tuple[tuple[str, ...], int] | None = None
        self._table: list[str] = []

    def owner(self, node: Node, key: bytes, mode: str) -> str:
        """Which node id should own ``key`` under this partitioning mode.

        HASH_MOD remaps most keys when membership changes; SLOTS moves about
        ``1/(N+1)``; KEY_RANGE keeps neighbors together; DIRECTORY honors pin;
        ITEM ignores pin. ``N`` is ``len(node.nodes)``.
        """
        partitioning = mode or node.last_partitioning
        nodes = list(node.nodes)
        if partitioning == "HASH_MOD": return nodes[_hash(key) % len(nodes)]
        if partitioning == "KEY_RANGE":
            first_byte = key[0] if key else 0
            return nodes[first_byte * len(nodes) // 256]
        if partitioning == "DIRECTORY":
            with node.lock: pinned = node.directory.get(key)
            if pinned in nodes: return pinned
            return self._slot_owner(node, key)
        if partitioning in ("SLOTS", "ITEM"): return self._slot_owner(node, key)
        return ""

    def on_add_node(self, node: Node, request: node_pb2.AddNodeRequest) -> node_pb2.StatusResponse:
        """Handle ``AddNode`` / pin. Return a StatusResponse. Move bytes that changed owner."""
        partitioning = mode(request.modes, "partitioning") or node.last_partitioning
        if partitioning: node.last_partitioning = partitioning
        if request.pin_key: return self._pin(node, request, partitioning)
        return self._add(node, request, partitioning)

    def rebalance(self, node: Node, modes: node_pb2.Modes) -> None:
        """Send keys this node holds to their current owner if that owner is someone else."""
        partitioning = mode(modes, "partitioning") or node.last_partitioning
        if not partitioning: return
        with node.lock: keys = list(node.store)
        self._send_to_owners(node, keys, partitioning)
        if partitioning == "DIRECTORY": self._pull_pinned(node)

    def _join_order_locked(self, node: Node) -> list[str]:
        """Members in the order they joined; caller holds self._lock."""
        if self._join_order is None: self._join_order = list(node.nodes)
        for n in node.nodes:
            if n not in self._join_order: self._join_order.append(n)
        return [n for n in self._join_order if n in node.nodes]

    def _slot_owner(self, node: Node, key: bytes) -> str:
        with self._lock:
            join_order = tuple(self._join_order_locked(node))
            if self._table_for != (join_order, node.n_slots):
                self._table = _slot_table(join_order, node.n_slots)
                self._table_for = (join_order, node.n_slots)
            return self._table[_hash(key) % node.n_slots]

    def _add(self, node: Node, request: node_pb2.AddNodeRequest, partitioning: str) -> node_pb2.StatusResponse:
        new_id = request.node_id
        if new_id == node.node_id:
            with self._lock:
                self._join_order_locked(node)
                self._join_order.remove(new_id)
                self._join_order.append(new_id)
            return node.status_now()

        with self._lock: self._join_order_locked(node)
        with node.lock: held = dict(node.store)
        old_owner = {k: self.owner(node, k, partitioning) for k in held} if partitioning else {}

        with node.lock:
            is_new = new_id not in node.nodes
            if is_new:
                node.nodes.append(new_id)
                node.nodes.sort(key=lambda x: (len(x), x))
            node.peers[new_id] = (request.host, int(request.port))
        if not is_new: return node.status_now()
        with self._lock:
            if new_id not in self._join_order: self._join_order.append(new_id)

        fwd = node_pb2.AddNodeRequest()
        fwd.CopyFrom(request)
        fwd.modes.partitioning = partitioning
        self._broadcast(node, fwd)
        self._send_pins(node, new_id)

        if partitioning:
            changed = [k for k in held if self.owner(node, k, partitioning) != old_owner[k]]
            print(f"[{node.node_id}] add-node {new_id} ({partitioning}): "
                  f"{len(changed)} of {len(held)} keys changed owner", flush=True)
            threading.Thread(
                target=self._send_to_owners, args=(node, changed, partitioning), daemon=True
            ).start()
        return node.status_now()

    def _pin(self, node: Node, request: node_pb2.AddNodeRequest, partitioning: str) -> node_pb2.StatusResponse:
        key, target = request.pin_key, request.node_id
        if partitioning != "DIRECTORY" or target not in node.nodes:
            status = node.status_now()
            status.ok = False
            return status

        with node.lock:
            already_pinned = node.directory.get(key) == target
            node.directory[key] = target
        if target != node.node_id: self._push(node, [key], target, partitioning)
        if not already_pinned:
            print(f"[{node.node_id}] pin {key!r} -> {target}", flush=True)
            fwd = node_pb2.AddNodeRequest()
            fwd.CopyFrom(request)
            fwd.modes.partitioning = "DIRECTORY"
            self._broadcast(node, fwd)
        return node.status_now()

    def _pull_pinned(self, node: Node) -> None:
        # Re-send the pin for keys pinned here that we don't have; the holder pushes them.
        with node.lock: missing = [k for k, t in node.directory.items() if t == node.node_id and k not in node.store]
        for key in missing:
            pin = node_pb2.AddNodeRequest(
                node_id=node.node_id, pin_key=key, modes=node_pb2.Modes(partitioning="DIRECTORY")
            )
            for peer in node.other_nodes():
                try: node.rpc(peer, "AddNode", pin, timeout_s=4.0)
                except RpcFailed: continue
                with node.lock:
                    if key in node.store: break

    def _send_to_owners(self, node: Node, keys: list[bytes], partitioning: str) -> None:
        keys_by_owner: dict[str, list[bytes]] = {}
        for key in keys:
            owner = self.owner(node, key, partitioning)
            if owner and owner != node.node_id: keys_by_owner.setdefault(owner, []).append(key)
        for owner, owned_keys in keys_by_owner.items(): self._push(node, owned_keys, owner, partitioning)

    def _push(self, node: Node, keys: list[bytes], target: str, partitioning: str) -> None:
        # Copy, don't move: write_completion keeps copies on every member.
        # Skip keys the target already has so an old copy can't overwrite a
        # newer value. Stop if the target is down; rebalance retries later.
        modes = node_pb2.Modes(partitioning=partitioning)
        for key in keys:
            with node.lock:
                if key not in node.store: continue
                value = node.store[key]
            try:
                existing = node.rpc(target, "Read", node_pb2.ReadRequest(key=key, modes=modes))
                if existing.ok and existing.found: continue
                node.rpc(target, "Replicate", node_pb2.ReplicateRequest(
                    key=key, value=value, kind="PARTITION_MOVE", index=-1, source=node.node_id, modes=modes,
                ))
            except RpcFailed: return

    def _broadcast(self, node: Node, request: node_pb2.AddNodeRequest) -> None:
        # One thread per peer so a paused peer can't push us past the client's 4 s timeout.
        def send(peer: str) -> None:
            try: node.rpc(peer, "AddNode", request, timeout_s=4.0)
            except RpcFailed: pass

        for peer in node.other_nodes(): threading.Thread(target=send, args=(peer,), daemon=True).start()

    def _send_pins(self, node: Node, new_id: str) -> None:
        with node.lock: pins = list(node.directory.items())

        def send() -> None:
            for key, target in pins:
                pin = node_pb2.AddNodeRequest(
                    node_id=target, pin_key=key, modes=node_pb2.Modes(partitioning="DIRECTORY")
                )
                try: node.rpc(new_id, "AddNode", pin, timeout_s=4.0)
                except RpcFailed: return

        threading.Thread(target=send, daemon=True).start()