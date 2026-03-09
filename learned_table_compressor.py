from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import math
import random
import struct
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


ByteKey = bytes
ValueTuple = Tuple[Any, ...]


def _common_prefix_len(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def _longest_common_prefix(values: Sequence[bytes]) -> bytes:
    if not values:
        return b""
    prefix = values[0]
    for item in values[1:]:
        prefix = prefix[: _common_prefix_len(prefix, item)]
        if not prefix:
            break
    return prefix


@dataclass(slots=True)
class ProbabilityTransformer:
    """
    Probability-aware transformation function F.

    The implementation follows the paper's spirit:
    1) Build byte-level conditional probabilities with additive smoothing.
    2) Map each key into a scalar in probability space through recursive accumulation.
    """

    alphabet_size: int = 257  # 0 is EOS, 1..256 map from raw byte+1
    _prefix_total: Dict[Tuple[int, ...], int] = field(init=False, default_factory=dict)
    _next_counts: Dict[Tuple[int, ...], List[int]] = field(init=False, default_factory=dict)
    _cum_prob_cache: Dict[Tuple[int, ...], List[float]] = field(init=False, default_factory=dict)

    @staticmethod
    def _to_symbols(key: bytes) -> List[int]:
        # Shift each byte to 1..256 and append EOS(0), so shorter keys remain ordered.
        return [b + 1 for b in key] + [0]

    def build(self, keys: Sequence[bytes]) -> None:
        prefix_total: Dict[Tuple[int, ...], int] = defaultdict(int)
        next_counts: Dict[Tuple[int, ...], List[int]] = {}
        alpha = self.alphabet_size

        for key in keys:
            symbols = self._to_symbols(key)
            prefix: Tuple[int, ...] = ()
            for sym in symbols:
                prefix_total[prefix] += 1
                arr = next_counts.get(prefix)
                if arr is None:
                    arr = [0] * alpha
                    next_counts[prefix] = arr
                arr[sym] += 1
                prefix = prefix + (sym,)

        self._prefix_total = dict(prefix_total)
        self._next_counts = next_counts
        self._cum_prob_cache.clear()

    def _prob(self, prefix: Tuple[int, ...], symbol: int) -> float:
        total = self._prefix_total.get(prefix, 0)
        arr = self._next_counts.get(prefix)
        count = 0 if arr is None else arr[symbol]
        return (1.0 + count) / (self.alphabet_size + total)

    def _cum_prob(self, prefix: Tuple[int, ...], symbol_exclusive: int) -> float:
        cached = self._cum_prob_cache.get(prefix)
        if cached is None:
            total = self._prefix_total.get(prefix, 0)
            arr = self._next_counts.get(prefix)
            denom = self.alphabet_size + total
            cached = [0.0] * (self.alphabet_size + 1)
            running = 0.0
            for sym in range(self.alphabet_size):
                count = 0 if arr is None else arr[sym]
                running += (1.0 + count) / denom
                cached[sym + 1] = running
            self._cum_prob_cache[prefix] = cached
        return cached[symbol_exclusive]

    def transform(self, key: bytes) -> float:
        symbols = self._to_symbols(key)
        prefix: Tuple[int, ...] = ()
        prefix_prob = 1.0
        result = 0.0
        for sym in symbols:
            result += prefix_prob * self._cum_prob(prefix, sym)
            prefix_prob *= self._prob(prefix, sym)
            prefix = prefix + (sym,)
        return result


@dataclass(slots=True)
class DiscretizationModel:
    split_points: List[float]

    @property
    def slot_count(self) -> int:
        return len(self.split_points) + 1

    def locate(self, transformed_value: float) -> int:
        return bisect_right(self.split_points, transformed_value)

    @classmethod
    def build(
        cls,
        sorted_keys: Sequence[bytes],
        transformed_values: Sequence[float],
        expected_entries: int,
    ) -> "DiscretizationModel":
        count = len(sorted_keys)
        if count <= 1:
            return cls(split_points=[])

        interval = max(1, math.ceil(count / max(1, expected_entries)))
        split_points: List[float] = []
        left = 0

        while left < count - 1:
            right_limit = min(left + interval - 1, count - 2)
            chosen: Optional[int] = None

            lcp = sorted_keys[left]
            for idx in range(left, right_limit + 1):
                if idx > left:
                    lcp = lcp[: _common_prefix_len(lcp, sorted_keys[idx])]

                if not lcp:
                    continue
                if idx == right_limit:
                    chosen = idx
                    break

                next_lcp_len = _common_prefix_len(lcp, sorted_keys[idx + 1])
                if next_lcp_len == 0:
                    chosen = idx
                    break

            if chosen is None:
                chosen = right_limit

            split_points.append(
                (transformed_values[chosen] + transformed_values[chosen + 1]) / 2.0
            )
            left = chosen + 1

        return cls(split_points=split_points)


@dataclass(slots=True)
class DataEntry:
    suffix: bytes
    value: Any


@dataclass(slots=True)
class NodeEntry:
    child: "TreeNode"


Entry = Union[DataEntry, NodeEntry, None]


@dataclass(slots=True)
class TreeNode:
    common_prefix: bytes
    transformer: ProbabilityTransformer
    model: DiscretizationModel
    entries: List[Entry]


class LearnedTreeGraph:
    """
    Deterministic learned tree-graph index:
    - node = common prefix + discretization model + entry list
    - entry = data entry or node entry
    """

    def __init__(self, expected_entries: int = 16):
        self.expected_entries = max(2, expected_entries)
        self.root: Optional[TreeNode] = None
        self._kv: Dict[bytes, Any] = {}

    def build_from_items(self, items: Iterable[Tuple[bytes, Any]]) -> None:
        kv: Dict[bytes, Any] = {}
        for key, value in items:
            if not isinstance(key, (bytes, bytearray)):
                raise TypeError("Index key must be bytes.")
            kv[bytes(key)] = value
        self._kv = kv
        ordered = sorted(self._kv.items(), key=lambda item: item[0])
        self.root = self._build_node(ordered)

    def _build_node(self, sorted_items: Sequence[Tuple[bytes, Any]]) -> Optional[TreeNode]:
        if not sorted_items:
            return None

        keys = [item[0] for item in sorted_items]
        prefix = _longest_common_prefix(keys)
        trimmed = [(key[len(prefix) :], value) for key, value in sorted_items]
        trimmed_keys = [item[0] for item in trimmed]

        transformer = ProbabilityTransformer()
        transformer.build(trimmed_keys)
        transformed = [transformer.transform(key) for key in trimmed_keys]
        model = DiscretizationModel.build(trimmed_keys, transformed, self.expected_entries)

        entries: List[Entry] = [None] * model.slot_count

        groups: List[Tuple[int, int, int]] = []  # (slot, begin, end)
        begin = 0
        total = len(trimmed)
        while begin < total:
            slot = model.locate(transformed[begin])
            end = begin + 1
            while end < total and model.locate(transformed[end]) == slot:
                end += 1
            groups.append((slot, begin, end))
            begin = end

        # Fallback split: avoid pathological one-slot recursion.
        if len(groups) == 1 and groups[0][1] == 0 and groups[0][2] == total and total > 1:
            mid = total // 2
            split = (transformed[mid - 1] + transformed[mid]) / 2.0
            model = DiscretizationModel([split])
            entries = [None, None]
            left_group = trimmed[:mid]
            right_group = trimmed[mid:]
            entries[0] = self._group_to_entry(left_group)
            entries[1] = self._group_to_entry(right_group)
            return TreeNode(prefix, transformer, model, entries)

        for slot, start, stop in groups:
            group = trimmed[start:stop]
            entries[slot] = self._group_to_entry(group)

        return TreeNode(prefix, transformer, model, entries)

    def _group_to_entry(self, group: Sequence[Tuple[bytes, Any]]) -> Entry:
        if not group:
            return None
        if len(group) == 1:
            key, value = group[0]
            return DataEntry(suffix=key, value=value)
        child = self._build_node(group)
        if child is None:
            return None
        return NodeEntry(child=child)

    def point_query(self, key: bytes) -> Optional[Any]:
        if self.root is None:
            return None
        return self._point_query_node(self.root, key)

    def _point_query_node(self, node: TreeNode, key: bytes) -> Optional[Any]:
        if not key.startswith(node.common_prefix):
            return None

        remaining = key[len(node.common_prefix) :]
        slot = node.model.locate(node.transformer.transform(remaining))
        if slot >= len(node.entries):
            return None
        entry = node.entries[slot]
        if entry is None:
            return None
        if isinstance(entry, DataEntry):
            return entry.value if entry.suffix == remaining else None
        return self._point_query_node(entry.child, remaining)

    def iter_items(self) -> Iterator[Tuple[bytes, Any]]:
        if self.root is None:
            return iter(())
        return self._iter_node(self.root, b"")

    def _iter_node(self, node: TreeNode, parent_prefix: bytes) -> Iterator[Tuple[bytes, Any]]:
        current_prefix = parent_prefix + node.common_prefix
        for entry in node.entries:
            if entry is None:
                continue
            if isinstance(entry, DataEntry):
                yield current_prefix + entry.suffix, entry.value
            else:
                yield from self._iter_node(entry.child, current_prefix)

    def range_query(self, lower: bytes, upper: bytes) -> List[Tuple[bytes, Any]]:
        if lower > upper:
            lower, upper = upper, lower
        results: List[Tuple[bytes, Any]] = []
        for key, value in self.iter_items():
            if key < lower:
                continue
            if key > upper:
                break
            results.append((key, value))
        return results

    def insert(self, key: bytes, value: Any) -> None:
        self._kv[key] = value
        self.build_from_items(self._kv.items())

    def delete(self, key: bytes) -> bool:
        if key not in self._kv:
            return False
        del self._kv[key]
        self.build_from_items(self._kv.items())
        return True

    def update(self, old_key: bytes, new_key: bytes, value: Any) -> None:
        if old_key != new_key:
            self._kv.pop(old_key, None)
        self._kv[new_key] = value
        self.build_from_items(self._kv.items())

    def __len__(self) -> int:
        return len(self._kv)

    def estimate_size_bytes(self, payload_sizer) -> int:
        if self.root is None:
            return 0
        return self._estimate_node(self.root, payload_sizer)

    def _estimate_node(self, node: TreeNode, payload_sizer) -> int:
        size = 4 + len(node.common_prefix)  # length marker + bytes
        size += 8 * len(node.model.split_points)  # doubles for division points

        for entry in node.entries:
            if entry is None:
                continue
            if isinstance(entry, DataEntry):
                size += 4 + len(entry.suffix)
                size += payload_sizer(entry.value)
            else:
                size += 8  # pointer
                size += self._estimate_node(entry.child, payload_sizer)
        return size


class PrimaryKeyCodec:
    def __init__(self, kind: str):
        if kind not in {"int", "float", "string", "bytes"}:
            raise ValueError(f"Unsupported primary key kind: {kind}")
        self.kind = kind

    @staticmethod
    def infer(sample: Any) -> "PrimaryKeyCodec":
        if isinstance(sample, bool):
            return PrimaryKeyCodec("int")
        if isinstance(sample, int):
            return PrimaryKeyCodec("int")
        if isinstance(sample, float):
            return PrimaryKeyCodec("float")
        if isinstance(sample, bytes):
            return PrimaryKeyCodec("bytes")
        return PrimaryKeyCodec("string")

    def encode(self, value: Any) -> bytes:
        if self.kind == "int":
            int_value = int(value)
            biased = int_value + (1 << 63)
            if biased < 0 or biased >= (1 << 64):
                raise OverflowError("Integer primary key out of supported int64 range.")
            return biased.to_bytes(8, "big", signed=False)

        if self.kind == "float":
            bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
            sortable = (~bits & ((1 << 64) - 1)) if (bits >> 63) else (bits ^ (1 << 63))
            return sortable.to_bytes(8, "big", signed=False)

        if self.kind == "bytes":
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError("Primary key must be bytes for bytes codec.")
            return bytes(value)

        return str(value).encode("utf-8")

    def decode(self, raw: bytes) -> Any:
        if self.kind == "int":
            return int.from_bytes(raw, "big", signed=False) - (1 << 63)

        if self.kind == "float":
            sortable = int.from_bytes(raw, "big", signed=False)
            bits = sortable ^ (1 << 63) if (sortable >> 63) else (~sortable & ((1 << 64) - 1))
            return struct.unpack(">d", bits.to_bytes(8, "big", signed=False))[0]

        if self.kind == "bytes":
            return raw

        return raw.decode("utf-8")


class LearnedTableCompressor:
    """
    Two-dimensional table compression with:
    - value arrays for byte-array fields
    - field-value mapping tree graphs
    - primary-key mapping tree graph
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        primary_key: str,
        expected_entries: int = 16,
        field_types: Optional[Dict[str, str]] = None,
    ):
        if not records:
            raise ValueError("records cannot be empty")

        self.primary_key = primary_key
        self.expected_entries = max(2, expected_entries)
        self.fields = [field for field in records[0].keys() if field != primary_key]
        if not self.fields:
            raise ValueError("table must contain at least one non-primary field")

        for row in records:
            if primary_key not in row:
                raise KeyError(f"Primary key '{primary_key}' missing in one record.")
            for field in self.fields:
                if field not in row:
                    raise KeyError(f"Field '{field}' missing in one record.")

        sample_pk = records[0][primary_key]
        self.pk_codec = PrimaryKeyCodec.infer(sample_pk)

        self.field_types = self._infer_field_types(records, field_types)

        self.value_arrays: Dict[str, List[Any]] = {}
        self.value_to_serial: Dict[str, Dict[bytes, int]] = {}
        self.field_indexes: Dict[str, LearnedTreeGraph] = {}

        self.primary_index = LearnedTreeGraph(expected_entries=self.expected_entries)

        self._build_value_arrays(records)
        self._build_primary_index(records)

    def _infer_field_types(
        self,
        records: Sequence[Dict[str, Any]],
        override: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for field in self.fields:
            if override and field in override:
                value = override[field]
                if value not in {"byte_array", "numeric"}:
                    raise ValueError(f"Invalid field type '{value}' for field '{field}'.")
                result[field] = value
                continue

            selected = None
            for row in records:
                candidate = row[field]
                if candidate is None:
                    continue
                selected = candidate
                break
            if isinstance(selected, (str, bytes, bytearray)) or selected is None:
                result[field] = "byte_array"
            else:
                result[field] = "numeric"
        return result

    @staticmethod
    def _as_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        return str(value).encode("utf-8")

    def _build_value_arrays(self, records: Sequence[Dict[str, Any]]) -> None:
        for field in self.fields:
            if self.field_types[field] != "byte_array":
                continue

            distinct: Dict[bytes, Any] = {}
            for row in records:
                original = row[field]
                raw = self._as_bytes(original)
                if raw not in distinct:
                    distinct[raw] = original

            sorted_pairs = sorted(distinct.items(), key=lambda item: item[0])
            value_array = [None] + [original for _, original in sorted_pairs]
            mapping: Dict[bytes, int] = {}
            for serial, (raw, _) in enumerate(sorted_pairs, start=1):
                mapping[raw] = serial

            index = LearnedTreeGraph(expected_entries=self.expected_entries)
            index.build_from_items((raw, serial) for raw, serial in mapping.items())

            self.value_arrays[field] = value_array
            self.value_to_serial[field] = mapping
            self.field_indexes[field] = index

    def _build_primary_index(self, records: Sequence[Dict[str, Any]]) -> None:
        pairs: List[Tuple[bytes, ValueTuple]] = []
        seen = set()
        for row in records:
            key = row[self.primary_key]
            raw_key = self.pk_codec.encode(key)
            if raw_key in seen:
                raise ValueError(f"Duplicate primary key detected: {key}")
            seen.add(raw_key)
            pairs.append((raw_key, self._encode_value_tuple(row, create_missing=False)))
        self.primary_index.build_from_items(pairs)

    def _get_or_create_serial(self, field: str, value: Any, create_missing: bool) -> int:
        raw = self._as_bytes(value)

        # Prefer learned field-value index lookup to stay consistent with the paper pipeline.
        serial = self.field_indexes[field].point_query(raw)
        if serial is not None:
            return int(serial)

        # Fallback fast-path map (kept for robustness).
        serial = self.value_to_serial[field].get(raw)
        if serial is not None:
            return serial

        if not create_missing:
            raise KeyError(f"Value '{value}' of field '{field}' not found in value array.")

        serial = len(self.value_arrays[field])
        self.value_arrays[field].append(value)
        self.value_to_serial[field][raw] = serial
        self.field_indexes[field].insert(raw, serial)
        return serial

    def _encode_value_tuple(self, row: Dict[str, Any], create_missing: bool) -> ValueTuple:
        encoded: List[Any] = []
        for field in self.fields:
            value = row[field]
            if self.field_types[field] == "byte_array":
                encoded.append(self._get_or_create_serial(field, value, create_missing))
            else:
                encoded.append(value)
        return tuple(encoded)

    def _decode_value_tuple(self, encoded: ValueTuple) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for idx, field in enumerate(self.fields):
            value = encoded[idx]
            if self.field_types[field] == "byte_array":
                serial = int(value)
                arr = self.value_arrays[field]
                if serial <= 0 or serial >= len(arr):
                    raise IndexError(f"Invalid serial {serial} for field '{field}'.")
                decoded[field] = arr[serial]
            else:
                decoded[field] = value
        return decoded

    def point_query(self, primary_value: Any) -> Optional[Dict[str, Any]]:
        raw_key = self.pk_codec.encode(primary_value)
        encoded_tuple = self.primary_index.point_query(raw_key)
        if encoded_tuple is None:
            return None
        result = {self.primary_key: self.pk_codec.decode(raw_key)}
        result.update(self._decode_value_tuple(encoded_tuple))
        return result

    def range_query(self, lower: Any, upper: Any) -> List[Dict[str, Any]]:
        lower_raw = self.pk_codec.encode(lower)
        upper_raw = self.pk_codec.encode(upper)
        rows: List[Dict[str, Any]] = []
        for raw_key, encoded_tuple in self.primary_index.range_query(lower_raw, upper_raw):
            row = {self.primary_key: self.pk_codec.decode(raw_key)}
            row.update(self._decode_value_tuple(encoded_tuple))
            rows.append(row)
        return rows

    def insert(self, record: Dict[str, Any]) -> bool:
        key_value = record[self.primary_key]
        raw_key = self.pk_codec.encode(key_value)
        existed = self.primary_index.point_query(raw_key) is not None
        encoded_tuple = self._encode_value_tuple(record, create_missing=True)
        self.primary_index.insert(raw_key, encoded_tuple)
        return not existed

    def delete(self, primary_value: Any) -> bool:
        raw_key = self.pk_codec.encode(primary_value)
        return self.primary_index.delete(raw_key)

    def update(self, old_primary: Any, new_record: Dict[str, Any]) -> None:
        new_primary = new_record[self.primary_key]
        old_key = self.pk_codec.encode(old_primary)
        new_key = self.pk_codec.encode(new_primary)
        new_encoded = self._encode_value_tuple(new_record, create_missing=True)
        self.primary_index.update(old_key, new_key, new_encoded)

    def all_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for raw_key, encoded_tuple in self.primary_index.iter_items():
            row = {self.primary_key: self.pk_codec.decode(raw_key)}
            row.update(self._decode_value_tuple(encoded_tuple))
            rows.append(row)
        return rows

    def estimate_compressed_size_bytes(self) -> int:
        total = 0

        for field in self.fields:
            if self.field_types[field] != "byte_array":
                continue
            # Value array payload
            for value in self.value_arrays[field][1:]:
                total += len(self._as_bytes(value))
            # Field index payload
            total += self.field_indexes[field].estimate_size_bytes(lambda _: 4)

        def primary_payload_size(encoded_tuple: ValueTuple) -> int:
            size = 0
            for idx, field in enumerate(self.fields):
                value = encoded_tuple[idx]
                if self.field_types[field] == "byte_array":
                    size += 4
                elif isinstance(value, float):
                    size += 8
                else:
                    size += 8 if isinstance(value, int) else len(self._as_bytes(value))
            return size

        total += self.primary_index.estimate_size_bytes(primary_payload_size)
        return total

    @staticmethod
    def estimate_original_size_bytes(records: Sequence[Dict[str, Any]], primary_key: str) -> int:
        total = 0
        for row in records:
            for field, value in row.items():
                if isinstance(value, float):
                    total += 8
                elif isinstance(value, int):
                    total += 8
                elif isinstance(value, (bytes, bytearray)):
                    total += len(value)
                else:
                    total += len(str(value).encode("utf-8"))
                if field == primary_key:
                    continue
        return total


def generate_mock_table(
    row_count: int = 2000,
    n_fields: int = 6,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """
    Generate a synthetic 2D table with one primary key and N fields.
    Fields are mixed: skewed strings + numeric attributes.
    """

    if n_fields < 1:
        raise ValueError("n_fields must be >= 1")

    rng = random.Random(seed)
    cities = ["beijing", "shanghai", "shenzhen", "guangzhou", "hangzhou", "nanjing"]
    departments = ["ai", "db", "cloud", "vision", "systems", "security"]
    levels = ["L1", "L2", "L3", "L4", "L5"]

    rows: List[Dict[str, Any]] = []
    for idx in range(1, row_count + 1):
        row: Dict[str, Any] = {"id": idx}

        for field_idx in range(1, n_fields + 1):
            name = f"f{field_idx}"
            if field_idx % 2 == 1:
                if field_idx % 3 == 1:
                    # Prefix-skewed strings, good for testing discretization robustness.
                    value = f"user_{idx % 200:03d}_{rng.choice(levels)}"
                elif field_idx % 3 == 2:
                    value = rng.choice(cities)
                else:
                    value = rng.choice(departments)
            else:
                if field_idx % 4 == 0:
                    value = round(rng.uniform(0, 100), 3)
                else:
                    value = rng.randint(0, 100_000)
            row[name] = value

        rows.append(row)

    return rows


def _records_to_map(records: Sequence[Dict[str, Any]], primary_key: str) -> Dict[Any, Dict[str, Any]]:
    return {row[primary_key]: row for row in records}


def run_correctness_validation() -> None:
    dataset = generate_mock_table(row_count=2500, n_fields=6, seed=42)
    compressor = LearnedTableCompressor(dataset, primary_key="id", expected_entries=24)

    baseline = _records_to_map(dataset, "id")

    # Full point-query validation.
    for pk, expected in baseline.items():
        actual = compressor.point_query(pk)
        if actual != expected:
            raise AssertionError(f"Point query mismatch for pk={pk}: {actual} != {expected}")

    rng = random.Random(123)

    # Random mixed operations with consistency checks.
    next_pk = max(baseline) + 1
    operations = 600

    for step in range(operations):
        roll = rng.random()

        if roll < 0.30:
            # Insert
            record = generate_mock_table(row_count=1, n_fields=6, seed=10000 + step)[0]
            record["id"] = next_pk
            next_pk += 1
            compressor.insert(record)
            baseline[record["id"]] = record

        elif roll < 0.55 and baseline:
            # Delete
            victim = rng.choice(list(baseline.keys()))
            deleted = compressor.delete(victim)
            if not deleted:
                raise AssertionError(f"Delete failed for existing key {victim}")
            del baseline[victim]

        elif roll < 0.80 and baseline:
            # Update (same key or key change)
            old_pk = rng.choice(list(baseline.keys()))
            updated = dict(baseline[old_pk])

            updated["f1"] = f"{updated['f1']}_u{step % 7}"
            updated["f2"] = int(updated["f2"]) + 17
            updated["f4"] = float(updated["f4"]) + 0.5

            change_key = rng.random() < 0.25
            if change_key:
                updated["id"] = next_pk
                next_pk += 1
                baseline.pop(old_pk)
                baseline[updated["id"]] = updated
            else:
                baseline[old_pk] = updated

            compressor.update(old_pk, updated)

        else:
            # Query checks
            if baseline:
                target = rng.choice(list(baseline.keys()))
                actual = compressor.point_query(target)
                if actual != baseline[target]:
                    raise AssertionError(f"Random point query mismatch on key {target}")

                low = rng.choice(list(baseline.keys()))
                high = rng.choice(list(baseline.keys()))
                lower, upper = (low, high) if low <= high else (high, low)
                expected_rows = [baseline[k] for k in sorted(baseline.keys()) if lower <= k <= upper]
                actual_rows = compressor.range_query(lower, upper)
                if actual_rows != expected_rows:
                    raise AssertionError(
                        "Range query mismatch "
                        f"for [{lower}, {upper}], expected={len(expected_rows)}, actual={len(actual_rows)}"
                    )

        # Periodic full snapshot check.
        if step % 50 == 0:
            compressed_rows = compressor.all_records()
            compressed_map = _records_to_map(compressed_rows, "id")
            if compressed_map != baseline:
                raise AssertionError(f"Snapshot mismatch at step={step}")

    original_size = LearnedTableCompressor.estimate_original_size_bytes(dataset, primary_key="id")
    compressed_size = compressor.estimate_compressed_size_bytes()
    ratio = (compressed_size / original_size) if original_size > 0 else 0.0

    print("Validation passed.")
    print(f"Initial rows: {len(dataset)}")
    print(f"Final rows: {len(baseline)}")
    print(f"Estimated original size: {original_size} bytes")
    print(f"Estimated compressed size: {compressed_size} bytes")
    print(f"Estimated compressed/original ratio: {ratio:.4f}")


if __name__ == "__main__":
    run_correctness_validation()
