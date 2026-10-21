"""Write/Read that arrived on a node that does not own the key.

In the readings: request routing — Designing Data-Intensive Applications ch. 7.

See week06/README.md. Useful imports: ``from platform import node_pb2``
and ``from platform.util import mode, RpcFailed, copy_with_hop``.

FORWARD passes the request along (hop + 1 each time, stop at 8), DIRECTORY_TIER
looks the owner up and sends once (one retry if the lookup was stale), and
CLIENT_AWARE just answers "not owner".
"""
from __future__ import annotations

from platform import node_pb2
from platform.node import Node
from platform.util import MAX_HOP, RpcFailed, copy_with_hop, hop_count, mode

from .base import PluginBase


class Plugin(PluginBase):
    def on_write(
        self, node: Node, request: node_pb2.WriteRequest, owner: str
    ) -> node_pb2.WriteResponse | None:
        """If this node should not store the Write, forward, look up, or reject.

        Return a WriteResponse, or None to mean “handle it here.” Increment
        ``request.hop`` when forwarding; stop after 8. A stale directory lookup
        should retry once rather than drop the bytes.
        """
        routing = mode(request.modes, "request_routing")
        if not routing or owner == node.node_id:
            return None

        if routing == "CLIENT_AWARE":
            return node_pb2.WriteResponse(ok=False, error="not owner", owner=owner)

        if hop_count(request) >= MAX_HOP:
            return node_pb2.WriteResponse(ok=False, error="hop limit", owner=owner)

        if routing == "FORWARD":
            return self._send_write(node, owner, copy_with_hop(request))

        if routing == "DIRECTORY_TIER":
            return self._directory_write(node, request, owner)

        return node_pb2.WriteResponse(
            ok=False, error=f"unknown request_routing {routing}", owner=owner
        )

    def on_read(
        self, node: Node, request: node_pb2.ReadRequest, owner: str
    ) -> node_pb2.ReadResponse | None:
        """Same idea for Read. Return a ReadResponse or None."""
        routing = mode(request.modes, "request_routing")
        if not routing or owner == node.node_id:
            return None

        if routing == "CLIENT_AWARE":
            return node_pb2.ReadResponse(ok=False, error="not owner", served_by=node.node_id)

        if hop_count(request) >= MAX_HOP:
            return node_pb2.ReadResponse(ok=False, error="hop limit", served_by=node.node_id)

        if routing == "FORWARD":
            return self._send_read(node, owner, copy_with_hop(request))

        if routing == "DIRECTORY_TIER":
            return self._directory_read(node, request, owner)

        return node_pb2.ReadResponse(
            ok=False, error=f"unknown request_routing {routing}", served_by=node.node_id
        )

    # ---- DIRECTORY_TIER ----

    def _lookup(self, node: Node, key: bytes, fallback: str) -> str:
        # node.directory is where pinned keys live (DIRECTORY partitioning).
        # Anything not pinned goes by the partitioning rule, which the node
        # already worked out and passed in as owner.
        with node.lock:
            pinned = node.directory.get(key)
        return pinned or fallback

    def _directory_write(self, node, request, owner):
        target = self._lookup(node, request.key, owner)
        if target == node.node_id:
            return None
        # Send as CLIENT_AWARE so the target either stores it (it really is
        # the owner) or answers "not owner" with who it thinks owns the key.
        # That answer is how we find out our lookup was stale.
        fwd = copy_with_hop(request)
        fwd.modes.request_routing = "CLIENT_AWARE"
        resp = self._send_write(node, target, fwd)
        if resp.ok or resp.error != "not owner" or not resp.owner or resp.owner == target:
            return resp

        # stale lookup: one retry at the owner the other node reported
        retry_target = resp.owner
        if retry_target == node.node_id:
            return None
        fwd = copy_with_hop(fwd)
        return self._send_write(node, retry_target, fwd)

    def _directory_read(self, node, request, owner):
        target = self._lookup(node, request.key, owner)
        if target == node.node_id:
            return None
        fwd = copy_with_hop(request)
        fwd.modes.request_routing = "CLIENT_AWARE"
        resp = self._send_read(node, target, fwd)
        if resp.ok or resp.error != "not owner":
            return resp
        # ReadResponse has no owner field, so the directory entry was the
        # stale part. Retry once at the owner from the partitioning rule.
        if owner == node.node_id:
            return None
        if owner == target:
            return resp
        return self._send_read(node, owner, copy_with_hop(fwd))

    # ---- sending ----

    def _send_write(self, node, target, fwd):
        try:
            return node.rpc(target, "Write", fwd)
        except RpcFailed:
            # owner is paused, dropped, or down. Say so instead of storing
            # the bytes on a node that doesn't own them.
            return node_pb2.WriteResponse(
                ok=False, error=f"forward to {target} failed", owner=target
            )

    def _send_read(self, node, target, fwd):
        try:
            return node.rpc(target, "Read", fwd)
        except RpcFailed:
            return node_pb2.ReadResponse(
                ok=False, error=f"forward to {target} failed", served_by=node.node_id
            )
