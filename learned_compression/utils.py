from __future__ import annotations

from typing import Iterable, Sequence


def ensure_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def lcp_two(a: bytes, b: bytes) -> bytes:
    upper = min(len(a), len(b))
    idx = 0
    while idx < upper and a[idx] == b[idx]:
        idx += 1
    return a[:idx]


def lcp_many(values: Sequence[bytes]) -> bytes:
    if not values:
        return b""
    prefix = values[0]
    for value in values[1:]:
        prefix = lcp_two(prefix, value)
        if not prefix:
            break
    return prefix


def lcp_slice(values: Sequence[bytes], left: int, right: int) -> bytes:
    if left > right:
        return b""
    return lcp_many(values[left : right + 1])


def estimate_scalar_size(value: object) -> int:
    if isinstance(value, (int, float)):
        return 4
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(str(value).encode("utf-8"))


def flatten_tuple_size(values: Iterable[object]) -> int:
    total = 0
    for item in values:
        total += estimate_scalar_size(item)
    return total
