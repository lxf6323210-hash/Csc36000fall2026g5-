
"""When Write returns: send copies, wait for W, time out if a node is paused.

In the readings: synchronous / asynchronous / hybrid (quorum) replication —
Think Distributed Systems §8.4.3; Database Internals ch. 11.

See week06/README.md. Useful imports: ``from platform import node_pb2``
and ``from platform.util import mode, RpcFailed``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from platform import node_pb2
from platform.node import Node
from platform.util import mode, RpcFailed

from .base import PluginBase



class Plugin(PluginBase):
    def spread(
        self,
        node: Node,
        key: bytes,
        value: bytes,
        owner: str,
        modes: node_pb2.Modes,
    ) -> node_pb2.WriteResponse:
        """Store a Write and wait for the required acknowledgments."""

        selected = mode(modes, "write_completion") or "ACK_ONE"
        n = len(node.nodes)

        if selected == "ACK_ALL":
            required = n
        elif selected == "ACK_QUORUM":
            required = n // 2 + 1
        else:
            required = 1

        # Store the Write locally and record it in the log.
        with node.lock:
            index = len(node.log)
            node.log.append((index, key, value))
            node.store[key] = value
            node.last_applied = index

        request = node_pb2.ReplicateRequest(
            key=key,
            value=value,
            kind="STATE",
            index=index,
            source=node.node_id,
            modes=modes,
        )

        acked = [node.node_id]
        peers = node.other_nodes()

        # Remember copies that still need acknowledgment.
        with node.lock:
            if not hasattr(node, "_write_hints"):
                node._write_hints = {}

            for peer in peers:
                node._write_hints[(peer, key)] = request

        def send(peer: str) -> tuple[str, bool]:
            try:
                response = node.rpc(peer, "Replicate", request)
                success = bool(response.ok)
            except RpcFailed:
                success = False

            if success:
                with node.lock:
                    hint_key = (peer, key)
                    if node._write_hints.get(hint_key) is request:
                        del node._write_hints[hint_key]

            return peer, success

        if not peers:
            return node_pb2.WriteResponse(
                ok=True,
                owner=owner,
                acked_by=acked,
            )

        # Send replication requests concurrently.
        executor = ThreadPoolExecutor(max_workers=len(peers))
        futures = [executor.submit(send, peer) for peer in peers]
        executor.shutdown(wait=False)

        # ACK_ONE succeeds immediately after the local Write.
        if len(acked) >= required:
            return node_pb2.WriteResponse(
                ok=True,
                owner=owner,
                acked_by=acked,
            )

        # Wait for enough acknowledgments or a timeout.
        try:
            for future in as_completed(futures, timeout=2.0):
                peer, success = future.result()

                if success:
                    acked.append(peer)

                if len(acked) >= required:
                    break
        except TimeoutError:
            pass

        if len(acked) >= required:
            return node_pb2.WriteResponse(
                ok=True,
                owner=owner,
                acked_by=acked,
            )

        return node_pb2.WriteResponse(
            ok=False,
            error="ack timeout",
            owner=owner,
            acked_by=acked,
        )

    def repair(
        self,
        node: Node,
        modes: node_pb2.Modes,
    ) -> None:
        """Retry Writes that failed to reach other nodes."""

        with node.lock:
            hints = list(
                getattr(node, "_write_hints", {}).items()
            )

        for (peer, key), request in hints:
            try:
                response = node.rpc(peer, "Replicate", request)
            except RpcFailed:
                continue

            if response.ok:
                with node.lock:
                    hint_key = (peer, key)
                    if node._write_hints.get(hint_key) is request:
                        del node._write_hints[hint_key]

    def on_read(
        self,
        node: Node,
        request: node_pb2.ReadRequest,
        owner: str,
    ) -> node_pb2.ReadResponse | None:
        """Repair a missing value by retrieving it from another node."""

        with node.lock:
            if request.key in node.store:
                return None

        # Try the owner first, then other replicas.
        peers = [owner] if owner and owner != node.node_id else []

        for peer in node.other_nodes():
            if peer not in peers:
                peers.append(peer)

        # Ask other nodes for the missing value.
        for peer in peers:
            try:
                response = node.rpc(peer, "Read", request)
            except RpcFailed:
                continue

            if not response.ok or not response.found:
                continue

            # Store the recovered value locally.
            with node.lock:
                if request.key not in node.store:
                    index = len(node.log)
                    node.log.append(
                        (index, request.key, response.value)
                    )
                    node.store[request.key] = response.value
                    node.last_applied = index

                value = node.store[request.key]

            return node_pb2.ReadResponse(
                ok=True,
                value=value,
                found=True,
                served_by=node.node_id,
            )

        return None
