"""Rules for two Reads after a Write on the same session.

In the readings: session models (read-own-writes, monotonic reads) —
Database Internals ch. 11.

See week06/README.md. Useful imports: ``from platform import node_pb2``
and ``from platform.util import mode, RpcFailed, copy_with_hop``.
"""
from __future__ import annotations

from platform import node_pb2
from platform.node import Node
from platform.util import RpcFailed, mode

from .base import PluginBase


def _key_version_locked(node: Node, key: bytes) -> int:
    """Return the newest log index for this key; caller holds node.lock."""
    indexes = [int(index) for index, logged_key, _value in node.log if logged_key == key]
    if indexes:
        return max(indexes)
    # Some supplied modes may populate the store before a key appears in the log.
    return int(node.last_applied) if key in node.store else -1


def _record_read_locked(node: Node, session_id: str, key: bytes, value: bytes, version: int) -> None:
    node.sessions_read.setdefault(session_id, {})[key] = value
    versions = getattr(node, "_session_read_versions", None)
    if versions is None:
        versions = {}
        node._session_read_versions = versions
    versions[(session_id, key)] = version


class Plugin(PluginBase):
    def on_write(self, node: Node, request: node_pb2.WriteRequest, value: bytes) -> None:
        """Remember the latest value/version written by each session for each key."""
        selected = mode(request.modes, "session_models")
        if selected not in ("READ_OWN_WRITES", "MONOTONIC_READS") or not request.session_id:
            return

        with node.lock:
            node.sessions_write.setdefault(request.session_id, {})[request.key] = value
            versions = getattr(node, "_session_write_versions", None)
            if versions is None:
                versions = {}
                node._session_write_versions = versions
            versions[(request.session_id, request.key)] = _key_version_locked(node, request.key)

    def on_read(
        self, node: Node, request: node_pb2.ReadRequest, owner: str
    ) -> node_pb2.ReadResponse | None:
        """Enforce read-own-writes and monotonic reads for a named session.

        Session reads are served by the key owner so a stale non-owner cannot
        return an older copy. The owner keeps the session's last-seen key version.
        """
        selected = mode(request.modes, "session_models")
        if selected not in ("READ_OWN_WRITES", "MONOTONIC_READS") or not request.session_id:
            return None

        # Keep session history with the key owner. This also lets a read on a
        # different node pull the session's latest value from that owner.
        if owner and owner != node.node_id:
            try:
                return node.rpc(owner, "Read", request)
            except RpcFailed:
                return node_pb2.ReadResponse(
                    ok=False,
                    error="session owner unavailable",
                    served_by=node.node_id,
                )

        session_id = request.session_id
        key = request.key
        with node.lock:
            found = key in node.store
            current_value = node.store.get(key, b"")
            current_version = _key_version_locked(node, key)

            previous_versions = getattr(node, "_session_read_versions", {})
            previous_version = previous_versions.get((session_id, key), -1)
            previous_value = node.sessions_read.get(session_id, {}).get(key)

            if selected == "READ_OWN_WRITES":
                own_value = node.sessions_write.get(session_id, {}).get(key)
                own_versions = getattr(node, "_session_write_versions", {})
                own_version = own_versions.get((session_id, key), -1)

                # If this node's copy predates the session's successful write,
                # return the value that the session wrote rather than stale bytes.
                if own_value is not None and (not found or current_version < own_version):
                    value = own_value
                    version = own_version
                    found = True
                else:
                    value = current_value
                    version = current_version

            else:  # MONOTONIC_READS
                # If this node's current version regressed, keep the session's
                # previously observed value instead of returning an older one.
                if previous_value is not None and current_version < previous_version:
                    value = previous_value
                    version = previous_version
                    found = True
                else:
                    value = current_value
                    version = current_version

            _record_read_locked(node, session_id, key, value, version)

        return node_pb2.ReadResponse(
            ok=True,
            value=value,
            found=found,
            served_by=node.node_id,
        )
