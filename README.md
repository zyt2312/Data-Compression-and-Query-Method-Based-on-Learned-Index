# Learned Table Compression (Paper-based)

This project implements an engineering version of the two-dimensional table compression scheme described in the paper *Data Compression and Query Method Based on Learned Index*, including:

- Probability transformation function `F` (based on byte-level conditional probabilities)
- Discretization model `M` (adaptive split points)
- Tree-graph learned index (common prefix + data entries / child node entries)
- 2D table encode/decode pipeline:
- value arrays
- field value mapping tree graphs
- primary key mapping tree graph
- Operation interfaces: point query, range query, insert, delete, update
- Randomized consistency validation script (CRUD + range checks against a baseline)

## Files

- `learned_table_compressor.py`: core implementation + mock data generator + validation entry point.

## Quick Start

```bash
python learned_table_compressor.py
```

When validation succeeds, it prints:

- Validation passed
- Initial/final row counts
- Estimated size before/after compression and compression ratio

## Use as a Module

```python
from learned_table_compressor import LearnedTableCompressor, generate_mock_table

rows = generate_mock_table(row_count=1000, n_fields=6, seed=42)
engine = LearnedTableCompressor(rows, primary_key="id", expected_entries=24)

row = engine.point_query(10)
rows_in_range = engine.range_query(10, 20)

engine.insert({"id": 5001, "f1": "user_x", "f2": 123, "f3": "ai", "f4": 10.5, "f5": "beijing", "f6": 999})
engine.update(5001, {"id": 5001, "f1": "user_y", "f2": 124, "f3": "db", "f4": 11.5, "f5": "shanghai", "f6": 1000})
engine.delete(5001)
```

## Notes

- This is a complete runnable implementation, with correctness and interface completeness as priorities.
- Current write operations (`insert`/`delete`/`update`) rebuild the index for consistency; for higher update throughput, it can be extended to a path-local subtree reconstruction strategy.