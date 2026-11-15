"""A minimal WebSocket client (RFC 6455) on the standard library, for Discord's gateway.

Text messages only, which is all the gateway needs. The frame functions are
also used by the fake gateway in the tests.
"""

import base64
import hashlib
import os
import socket
import ssl
import struct
from urllib.parse import urlparse

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
TEXT, CLOSE, PING, PONG, CONTINUATION = 0x1, 0x8, 0x9, 0xA, 0x0


def accept_key(key):
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def encode(opcode, payload, mask=True):
    header = bytes([0x80 | opcode])
    length = len(payload)
    flag = 0x80 if mask else 0
    if length < 126:
        header += bytes([flag | length])
    elif length < 65536:
        header += bytes([flag | 126]) + struct.pack("!H", length)
    else:
        header += bytes([flag | 127]) + struct.pack("!Q", length)
    if not mask:
        return header + payload
    key = os.urandom(4)
    return header + key + bytes(byte ^ key[i % 4] for i, byte in enumerate(payload))


def read_exactly(receive, count):
    data = b""
    while len(data) < count:
        chunk = receive(count - len(data))
        if not chunk:
            raise ConnectionError("the connection closed")
        data += chunk
    return data


def read_frame(receive):
    """(final, opcode, payload) of the next frame. `receive(count)` reads bytes, like socket.recv."""
    first, second = read_exactly(receive, 2)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exactly(receive, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exactly(receive, 8))[0]
    key = read_exactly(receive, 4) if second & 0x80 else None
    payload = read_exactly(receive, length)
    if key:
        payload = bytes(byte ^ key[i % 4] for i, byte in enumerate(payload))
    return bool(first & 0x80), first & 0x0F, payload


class WebSocket:
    def __init__(self, url, timeout=60):
        parts = urlparse(url)
        secure = parts.scheme == "wss"
        port = parts.port or (443 if secure else 80)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        sock = socket.create_connection((parts.hostname, port), timeout=timeout)
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((
            f"GET {path} HTTP/1.1\r\nHost: {parts.hostname}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("the server closed the connection during the handshake")
            response += chunk
        head, self.leftover = response.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n")[0].decode(errors="replace")
        if " 101 " not in status or accept_key(key).encode() not in head:
            raise ConnectionError(f"WebSocket handshake failed: {status}")
        self.sock = sock

    def send(self, text):
        self.sock.sendall(encode(TEXT, text.encode()))

    def recv(self):
        """The next text message, or None when the server closes the connection."""
        message = b""
        while True:
            final, opcode, payload = read_frame(self._receive)
            if opcode == PING:
                self.sock.sendall(encode(PONG, payload))
            elif opcode == CLOSE:
                return None
            elif opcode in (TEXT, CONTINUATION):
                message += payload
                if final:
                    return message.decode()

    def _receive(self, count):
        if self.leftover:  # bytes that arrived with the handshake
            data, self.leftover = self.leftover[:count], self.leftover[count:]
            return data
        return self.sock.recv(count)

    def close(self):
        try:
            self.sock.sendall(encode(CLOSE, b""))
        except OSError:
            pass
        self.sock.close()
