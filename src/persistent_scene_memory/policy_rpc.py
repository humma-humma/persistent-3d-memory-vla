"""Small loopback RPC primitives shared by split simulator/policy environments."""

from __future__ import annotations

import pickle
import socket
import struct


_HEADER = struct.Struct("!Q")


def send_message(connection: socket.socket, value: object) -> None:
    payload = pickle.dumps(value, protocol=5)
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def receive_message(connection: socket.socket) -> object:
    header = _receive_exact(connection, _HEADER.size)
    size = _HEADER.unpack(header)[0]
    if size > 256 * 1024 * 1024:
        raise ValueError("RPC payload exceeds 256 MiB")
    return pickle.loads(_receive_exact(connection, size))


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("RPC connection closed during a message")
        chunks.extend(chunk)
    return bytes(chunks)
