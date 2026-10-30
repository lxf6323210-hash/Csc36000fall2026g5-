"""Who may accept a Write on this node, especially after a split.

In the readings: single-leader, multileader, leaderless — Think Distributed
Systems §8.4.5.

See week06/README.md. Useful imports: ``from platform.util import mode, copy_with_hop``.
"""
from __future__ import annotations

from platform import node_pb2
from platform.node import Node
from platform.util import MAX_HOP, RpcFailed, copy_with_hop, hop_count, mode

from .base import PluginBase


class Plugin(PluginBase):
    def admit(
        self, node: Node, request: node_pb2.WriteRequest, owner: str
    ) -> node_pb2.WriteResponse | None:
        """Drive the Write: accept here (return None), forward, or reject.

        ``None`` means this node continues and stores. A WriteResponse means
        the plugin already forwarded or failed (``not leader``, hop limit, …).
        """
        rep_mode = mode(request.modes, "leader_replication")

        # If no leader_replication mode is specified in the request, raise
        # NotImplementedError so node._try() returns MISSING and falls back
        # to the default owner / request_routing checks in node.py.
        if not rep_mode:
            raise NotImplementedError("leader_replication mode not active")

        # If this node was isolated via `fault --isolate on`, its drop_to set
        # blocks all outbound peers (len(node.reachable()) <= 1). Any request
        # forwarded from another peer (hop > 0) must be rejected because this
        # node is disconnected from the rest of the cluster.
        if (
            hop_count(request) > 0
            and len(node.other_nodes()) > 0
            and len(node.reachable()) <= 1
        ):
            return node_pb2.WriteResponse(
                ok=False, error="not leader", owner=node.leader or node.node_id
            )

        if rep_mode == "SINGLE_LEADER":
            # If there is no known leader in the current view, reject the write.
            if not node.leader:
                return node_pb2.WriteResponse(
                    ok=False, error="not leader", owner=""
                )

            # If this node is the leader, admit the write for local storage.
            if node.node_id == node.leader:
                return None

            # Otherwise, this node is a follower. Check hop limit before forwarding.
            if hop_count(request) >= MAX_HOP:
                return node_pb2.WriteResponse(
                    ok=False, error="hop limit", owner=node.leader
                )

            # If the leader is known to be unreachable from this follower, reject immediately.
            if node.leader not in node.reachable():
                return node_pb2.WriteResponse(
                    ok=False, error="not leader", owner=node.leader
                )

            # Forward the Write request to the leader node.
            try:
                fwd_request = copy_with_hop(request)
                return node.rpc(node.leader, "Write", fwd_request)
            except RpcFailed:
                # If the RPC to the leader times out, is dropped, or fails,
                # reject the write with "not leader".
                return node_pb2.WriteResponse(
                    ok=False, error="not leader", owner=node.leader
                )

        if rep_mode == "LEADERLESS":
            # In leaderless mode, any node can accept a Write directly, even
            # during a network split.
            return None

        if rep_mode == "MULTI_LEADER":
            # In multi-leader mode, multiple nodes may accept Writes locally.
            # Replication to other reachable leaders happens in after_local_write().
            return None

        raise NotImplementedError(f"unknown leader_replication mode: {rep_mode}")

    def after_local_write(
        self,
        node: Node,
        request: node_pb2.WriteRequest,
        value: bytes,
        owner: str,
    ) -> None:
        """After a successful local Write: MULTI_LEADER copies to other leaders."""
        rep_mode = mode(request.modes, "leader_replication")

        if rep_mode != "MULTI_LEADER":
            return

        # Only the coordinator node that originally accepted the client's Write
        # (hop == 0) broadcasts the copy to other reachable leaders. This prevents
        # infinite replication loops between leaders.
        if hop_count(request) > 0:
            return

        # If this node is completely isolated, it cannot reach any other leaders.
        if len(node.other_nodes()) > 0 and len(node.reachable()) <= 1:
            return

        fwd_request = copy_with_hop(request)
        fwd_request.value = value

        for peer in node.other_nodes():
            if peer in node.reachable():
                try:
                    node.rpc(peer, "Write", fwd_request)
                except RpcFailed:
                    # Ignore unreachable or paused peers during a network split
                    # so the local write still succeeds.
                    pass
