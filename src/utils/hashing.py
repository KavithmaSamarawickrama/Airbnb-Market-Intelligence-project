"""
Hashing utilities for file integrity verification and deduplication.

Computes streaming SHA256 hashes during file download.
"""

import hashlib
from typing import BinaryIO


def compute_streaming_sha256(stream: BinaryIO, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 hash from a file-like object without loading it all into RAM.

    Args:
        stream: File-like object opened in binary mode
        chunk_size: Read chunk size in bytes (default 64KB)

    Returns:
        Hex-encoded SHA256 hash string
    """
    sha256 = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        sha256.update(chunk)
    return sha256.hexdigest()


def compute_sha256(data: bytes) -> str:
    """
    Compute SHA256 hash of bytes.

    Args:
        data: Binary data

    Returns:
        Hex-encoded SHA256 hash string
    """
    return hashlib.sha256(data).hexdigest()
