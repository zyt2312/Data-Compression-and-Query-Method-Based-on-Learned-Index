from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ProbabilityTransform:
    """Byte-array to probability-space mapper defined by the paper."""

    first_count: dict[int, int] = field(default_factory=dict)
    second_count: dict[tuple[int, int], int] = field(default_factory=dict)
    prefix_cache: dict[bytes, tuple[float, float]] = field(default_factory=dict)

    @staticmethod
    def _h(prefix: bytes) -> int:
        if not prefix:
            return -1
        return 256 * prefix[0] + prefix[-1]

    @classmethod
    def from_keys(cls, keys: list[bytes]) -> "ProbabilityTransform":
        first = defaultdict(int)
        second = defaultdict(int)

        for key in keys:
            for k in range(len(key)):
                prefix = key[:k]
                h_val = cls._h(prefix)
                next_byte = key[k]
                first[h_val] += 1
                second[(h_val, next_byte)] += 1

        return cls(first_count=dict(first), second_count=dict(second))

    def conditional_probability(self, prefix: bytes, b: int) -> float:
        h_val = self._h(prefix)
        n1 = self.first_count.get(h_val, 0)
        n2 = self.second_count.get((h_val, b), 0)
        return (1.0 + n2) / (256.0 + n1)

    def prefix_probability_and_f(self, prefix: bytes) -> tuple[float, float]:
        if prefix in self.prefix_cache:
            return self.prefix_cache[prefix]

        prob = 1.0
        f_value = 0.0
        built = b""
        for byte_value in prefix:
            mass = 0.0
            for b in range(byte_value):
                mass += self.conditional_probability(built, b)
            f_value += prob * mass
            prob *= self.conditional_probability(built, byte_value)
            built += bytes([byte_value])

        self.prefix_cache[prefix] = (prob, f_value)
        return prob, f_value

    def transform(self, key: bytes) -> float:
        _, f_value = self.prefix_probability_and_f(key)
        return f_value
