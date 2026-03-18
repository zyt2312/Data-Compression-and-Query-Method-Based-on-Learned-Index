from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import ceil
from typing import Generic, Iterable, Iterator, TypeVar, Union

from .prob_transform import ProbabilityTransform
from .utils import lcp_slice, lcp_two

T = TypeVar("T")


@dataclass
class DiscretizationModel:
    boundaries: list[float]

    def predict(self, x: float) -> int:
        return bisect_right(self.boundaries, x)

    @property
    def size(self) -> int:
        return len(self.boundaries) + 1


@dataclass
class DataEntry(Generic[T]):
    suffix: bytes
    value: T


@dataclass
class NodeEntry(Generic[T]):
    child: "TreeNode[T]"


Entry = Union[DataEntry[T], NodeEntry[T]]


@dataclass
class TreeNode(Generic[T]):
    common_prefix: bytes
    model: DiscretizationModel
    transform: ProbabilityTransform
    entries: dict[int, Entry[T]] = field(default_factory=dict)

    def find(self, key: bytes) -> T | None:
        if not key.startswith(self.common_prefix):
            return None

        rest = key[len(self.common_prefix) :]
        slot = self.model.predict(self.transform.transform(rest))
        entry = self.entries.get(slot)
        if entry is None:
            return None
        if isinstance(entry, DataEntry):
            if entry.suffix == rest:
                return entry.value
            return None
        return entry.child.find(rest)

    def iter_items(self, path_prefix: bytes = b"") -> Iterator[tuple[bytes, T]]:
        local_prefix = path_prefix + self.common_prefix
        for slot in sorted(self.entries.keys()):
            entry = self.entries[slot]
            if isinstance(entry, DataEntry):
                yield (local_prefix + entry.suffix, entry.value)
            else:
                yield from entry.child.iter_items(local_prefix)


class LearnedTreeIndex(Generic[T]):
    def __init__(self, expected_entries: int = 32) -> None:
        self.expected_entries = max(2, expected_entries)
        self._data: dict[bytes, T] = {}
        self.root: TreeNode[T] | None = None

    @staticmethod
    def _build_discretization_model(
        keys: list[bytes],
        transform: ProbabilityTransform,
        expected_entries: int,
    ) -> DiscretizationModel:
        n = len(keys)
        if n <= 1:
            return DiscretizationModel([])

        step = max(1, ceil(n / expected_entries))
        boundaries: list[float] = []
        left = 0

        while left < n - 1:
            right_limit = min(n - 1, left + step - 1)
            chosen = right_limit

            for probe in range(left, right_limit + 1):
                lcp_non_empty = len(lcp_slice(keys, left, probe)) > 0
                if not lcp_non_empty:
                    continue

                is_stop = probe == right_limit
                if probe < n - 1:
                    next_lcp_empty = len(lcp_slice(keys, left, probe + 1)) == 0
                    is_stop = is_stop or next_lcp_empty

                if is_stop:
                    chosen = probe
                    break

            if chosen >= n - 1:
                break

            left_f = transform.transform(keys[chosen])
            right_f = transform.transform(keys[chosen + 1])
            boundaries.append((left_f + right_f) / 2.0)
            left = chosen + 1

        return DiscretizationModel(boundaries)

    @classmethod
    def _build_node(
        cls,
        pairs: list[tuple[bytes, T]],
        expected_entries: int,
    ) -> TreeNode[T]:
        keys = [k for k, _ in pairs]
        common_prefix = keys[0]
        for key in keys[1:]:
            common_prefix = lcp_two(common_prefix, key)
            if not common_prefix:
                break

        trimmed_pairs = [(k[len(common_prefix) :], v) for k, v in pairs]
        trimmed_keys = [k for k, _ in trimmed_pairs]

        transform = ProbabilityTransform.from_keys(trimmed_keys)
        model = cls._build_discretization_model(trimmed_keys, transform, expected_entries)

        groups: dict[int, list[tuple[bytes, T]]] = {}
        for key, value in trimmed_pairs:
            slot = model.predict(transform.transform(key))
            groups.setdefault(slot, []).append((key, value))

        entries: dict[int, Entry[T]] = {}
        for slot, group in groups.items():
            if len(group) == 1:
                entries[slot] = DataEntry(suffix=group[0][0], value=group[0][1])
            else:
                child_pairs = sorted(group, key=lambda item: item[0])
                entries[slot] = NodeEntry(child=cls._build_node(child_pairs, expected_entries))

        return TreeNode(
            common_prefix=common_prefix,
            model=model,
            transform=transform,
            entries=entries,
        )

    def bulk_load(self, items: Iterable[tuple[bytes, T]]) -> None:
        self._data = dict(items)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._data:
            self.root = None
            return
        pairs = sorted(self._data.items(), key=lambda item: item[0])
        self.root = self._build_node(pairs, self.expected_entries)

    def find(self, key: bytes) -> T | None:
        if self.root is None:
            return None
        return self.root.find(key)

    def insert(self, key: bytes, value: T) -> None:
        self._data[key] = value
        self._rebuild()

    def delete(self, key: bytes) -> bool:
        if key not in self._data:
            return False
        del self._data[key]
        self._rebuild()
        return True

    def update(self, key: bytes, value: T) -> bool:
        if key not in self._data:
            return False
        self._data[key] = value
        self._rebuild()
        return True

    def items(self) -> list[tuple[bytes, T]]:
        if self.root is None:
            return []
        return list(self.root.iter_items())

    def range_query(self, low: bytes, high: bytes) -> list[tuple[bytes, T]]:
        if low > high:
            low, high = high, low
        return [(k, v) for k, v in self.items() if low <= k <= high]

    def __len__(self) -> int:
        return len(self._data)
