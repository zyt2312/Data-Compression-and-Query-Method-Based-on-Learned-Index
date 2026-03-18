from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tree_index import DataEntry, LearnedTreeIndex, NodeEntry, TreeNode
from .utils import estimate_scalar_size


def encode_str(value: str) -> bytes:
    return value.encode("utf-8")


def encode_pk(value: Any) -> bytes:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("This implementation requires non-negative integer or string primary keys")
        return f"{value:020d}".encode("ascii")
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


@dataclass
class CompressionReport:
    original_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes == 0:
            return 1.0
        return self.compressed_bytes / self.original_bytes


@dataclass
class MemoryBreakdown:
    model_bytes: int = 0
    node_prefix_bytes: int = 0
    data_suffix_bytes: int = 0
    node_pointer_bytes: int = 0
    data_value_bytes: int = 0

    @property
    def total(self) -> int:
        return (
            self.model_bytes
            + self.node_prefix_bytes
            + self.data_suffix_bytes
            + self.node_pointer_bytes
            + self.data_value_bytes
        )


class LearnedTableCompressor:
    """Paper-style table compression with learned tree-graph indexes."""

    FLOAT_BYTES = 4
    INT_BYTES = 4
    LENGTH_MARK_BYTES = 4
    POINTER_BYTES = 4
    ENCODED_TUPLE_BYTES = 1

    def __init__(
        self,
        primary_key: str,
        string_fields: list[str],
        expected_entries: int = 32,
    ) -> None:
        self.primary_key = primary_key
        self.string_fields = list(string_fields)
        self.expected_entries = expected_entries

        self.value_arrays: dict[str, list[str]] = {}
        self.field_indexes: dict[str, LearnedTreeIndex[int]] = {}
        self.pk_index: LearnedTreeIndex[tuple[Any, ...]] = LearnedTreeIndex(expected_entries)
        self.non_pk_fields: list[str] = []

    def fit(self, records: list[dict[str, Any]]) -> None:
        if not records:
            self.value_arrays = {}
            self.field_indexes = {}
            self.pk_index = LearnedTreeIndex(self.expected_entries)
            self.non_pk_fields = []
            return

        self.non_pk_fields = [k for k in records[0].keys() if k != self.primary_key]

        self.value_arrays = {}
        self.field_indexes = {}
        for field in self.string_fields:
            unique_values = sorted({str(record[field]) for record in records})
            self.value_arrays[field] = unique_values
            field_index = LearnedTreeIndex[int](self.expected_entries)
            field_index.bulk_load(
                (encode_str(v), idx) for idx, v in enumerate(unique_values)
            )
            self.field_indexes[field] = field_index

        pk_items: list[tuple[bytes, tuple[Any, ...]]] = []
        for record in records:
            pk = record[self.primary_key]
            encoded_tuple = self._encode_non_pk_tuple(record)
            pk_items.append((encode_pk(pk), encoded_tuple))
        self.pk_index.bulk_load(pk_items)

    def _encode_non_pk_tuple(self, record: dict[str, Any]) -> tuple[Any, ...]:
        encoded: list[Any] = []
        for field in self.non_pk_fields:
            value = record[field]
            if field in self.string_fields:
                string_value = str(value)
                idx = self.field_indexes[field].find(encode_str(string_value))
                if idx is None:
                    self._append_field_value(field, string_value)
                    idx = self.field_indexes[field].find(encode_str(string_value))
                    if idx is None:
                        raise RuntimeError(f"Field value index build failed: {field}={string_value}")
                encoded.append(idx)
            else:
                encoded.append(value)
        return tuple(encoded)

    def _append_field_value(self, field: str, value: str) -> int:
        if field not in self.value_arrays:
            self.value_arrays[field] = []
            self.field_indexes[field] = LearnedTreeIndex(self.expected_entries)

        idx = len(self.value_arrays[field])
        self.value_arrays[field].append(value)
        self.field_indexes[field].insert(encode_str(value), idx)
        return idx

    def _decode_tuple(self, encoded_values: tuple[Any, ...]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for field, value in zip(self.non_pk_fields, encoded_values):
            if field in self.string_fields:
                decoded[field] = self.value_arrays[field][int(value)]
            else:
                decoded[field] = value
        return decoded

    def point_query(self, pk_value: Any) -> dict[str, Any] | None:
        encoded_tuple = self.pk_index.find(encode_pk(pk_value))
        if encoded_tuple is None:
            return None
        decoded = self._decode_tuple(encoded_tuple)
        decoded[self.primary_key] = pk_value
        return decoded

    def range_query(self, low_pk: Any, high_pk: Any) -> list[dict[str, Any]]:
        kvs = self.pk_index.range_query(encode_pk(low_pk), encode_pk(high_pk))
        result: list[dict[str, Any]] = []
        for key_bytes, encoded_tuple in kvs:
            pk = key_bytes.decode("ascii") if isinstance(low_pk, int) else key_bytes.decode("utf-8")
            if isinstance(low_pk, int):
                pk = int(pk)
            row = self._decode_tuple(encoded_tuple)
            row[self.primary_key] = pk
            result.append(row)
        return result

    def insert(self, record: dict[str, Any]) -> None:
        pk = record[self.primary_key]
        encoded_tuple = self._encode_non_pk_tuple(record)
        self.pk_index.insert(encode_pk(pk), encoded_tuple)

    def delete(self, pk_value: Any) -> bool:
        return self.pk_index.delete(encode_pk(pk_value))

    def update(self, old_pk: Any, new_record: dict[str, Any]) -> None:
        new_pk = new_record[self.primary_key]
        if old_pk != new_pk:
            self.insert(new_record)
            self.delete(old_pk)
            return

        encoded_tuple = self._encode_non_pk_tuple(new_record)
        updated = self.pk_index.update(encode_pk(new_pk), encoded_tuple)
        if not updated:
            self.pk_index.insert(encode_pk(new_pk), encoded_tuple)

    def estimate_memory(self, original_records: list[dict[str, Any]]) -> CompressionReport:
        original_bytes = 0
        for row in original_records:
            for value in row.values():
                original_bytes += estimate_scalar_size(value)

        compressed_bytes = self._estimate_value_arrays_bytes()
        compressed_bytes += self._estimate_index_bytes(self.pk_index, is_primary_index=True)
        for index in self.field_indexes.values():
            compressed_bytes += self._estimate_index_bytes(index, is_primary_index=False)

        return CompressionReport(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )

    def _estimate_value_arrays_bytes(self) -> int:
        total = 0
        for values in self.value_arrays.values():
            for value in values:
                total += len(value.encode("utf-8"))
        return total

    def _estimate_index_bytes(self, index: LearnedTreeIndex[Any], is_primary_index: bool) -> int:
        if index.root is None:
            return 0
        breakdown = self._walk_node_memory(index.root, is_primary_index)
        return breakdown.total

    def _walk_node_memory(self, node: TreeNode[Any], is_primary_index: bool) -> MemoryBreakdown:
        breakdown = MemoryBreakdown()

        breakdown.model_bytes += len(node.model.boundaries) * self.FLOAT_BYTES
        breakdown.node_prefix_bytes += self.LENGTH_MARK_BYTES + len(node.common_prefix)

        for entry in node.entries.values():
            if isinstance(entry, DataEntry):
                breakdown.data_suffix_bytes += self.LENGTH_MARK_BYTES + len(entry.suffix)
                if is_primary_index:
                    breakdown.data_value_bytes += self.ENCODED_TUPLE_BYTES
                else:
                    breakdown.data_value_bytes += self.INT_BYTES
            elif isinstance(entry, NodeEntry):
                breakdown.node_pointer_bytes += self.POINTER_BYTES
                child_breakdown = self._walk_node_memory(entry.child, is_primary_index)
                breakdown.model_bytes += child_breakdown.model_bytes
                breakdown.node_prefix_bytes += child_breakdown.node_prefix_bytes
                breakdown.data_suffix_bytes += child_breakdown.data_suffix_bytes
                breakdown.node_pointer_bytes += child_breakdown.node_pointer_bytes
                breakdown.data_value_bytes += child_breakdown.data_value_bytes

        return breakdown
