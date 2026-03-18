# Learned-Index-Based Data Compression and Query (Python)

This repository implements the method described in the paper *Data Compression and Query Method Based on Learned Index*.

## Implemented Features

- Byte-level conditional-probability transform function `F`
- Adaptive discretization model `M`
- Tree-graph learned index (common prefix + data entries + child-node entries)
- Table compression with:
  - value arrays
  - field-value learned mapping trees
  - primary-key learned mapping tree
- Query without decompression:
  - point query
  - range query
- Data operations:
  - insert
  - delete
  - update
- Post-compression memory estimation with paper-style component accounting

## Project Structure

```text
learning_index_finish/
├─ learned_compression/
│  ├─ __init__.py
│  ├─ prob_transform.py
│  ├─ tree_index.py
│  ├─ table_compressor.py
│  ├─ synthetic_data.py
│  └─ utils.py
├─ demo.py
└─ README.markdown
```

## Requirements

- Python 3.8+
- No third-party dependencies

## Run Demo

```bash
python demo.py
```

The demo executes:

1. synthetic table generation
2. index construction
3. point query
4. range query
5. insert
6. update
7. delete
8. compression statistics

## Data Format

Input data is `list[dict]`, one dictionary per row.

```python
rows = [
    {"id": 1, "name": "Alice", "city": "Shenzhen", "category": "A", "price": 10.5, "quantity": 2},
    {"id": 2, "name": "Bob", "city": "Guangzhou", "category": "B", "price": 20.0, "quantity": 5},
]
```

## Usage

```python
from learned_compression import LearnedTableCompressor

compressor = LearnedTableCompressor(
    primary_key="id",
    string_fields=["name", "city", "category"],
    expected_entries=32,
)

compressor.fit(rows)

row = compressor.point_query(100)
rows_in_range = compressor.range_query(100, 200)

compressor.insert(new_record)
compressor.update(old_pk=100, new_record=updated_record)
compressor.delete(100)

report = compressor.estimate_memory(rows)
print(report.original_bytes, report.compressed_bytes, report.compression_ratio)
```

## Memory Accounting Rules (Current Implementation)

`estimate_memory` computes:

- **Before compression**: per-cell scalar size accumulation.
- **After compression**:
  - value arrays
  - index model boundaries (`float`, 4 bytes each)
  - node common prefixes (`length marker + bytes`)
  - data-entry suffixes (`length marker + bytes`)
  - child-node pointers (`4 bytes` each)
  - primary-key data-entry encoded tuple (`fixed 1`)
  - field-index data-entry serial number (`int`, 4 bytes)

Constants used:

- `FLOAT_BYTES = 4`
- `INT_BYTES = 4`
- `LENGTH_MARK_BYTES = 4`
- `POINTER_BYTES = 4`
- `ENCODED_TUPLE_BYTES = 1`

## Mapping to Paper Sections

- Value Array: `LearnedTableCompressor.value_arrays`
- Transform Function `F`: `ProbabilityTransform`
- Discretization Model `M`: `DiscretizationModel`
- Index Construction: `LearnedTreeIndex._build_node`
- Encode/Decode: `_encode_non_pk_tuple`, `_decode_tuple`
- Point Query / Range Query: `point_query`, `range_query`
- Insert / Delete / Update: `insert`, `delete`, `update`
