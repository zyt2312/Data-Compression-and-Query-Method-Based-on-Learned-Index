from __future__ import annotations

from pprint import pprint

from learned_compression import LearnedTableCompressor
from learned_compression.synthetic_data import generate_table


def main() -> None:
    rows = generate_table(num_rows=3000, seed=20260317)

    compressor = LearnedTableCompressor(
        primary_key="id",
        string_fields=["name", "city", "category"],
        expected_entries=32,
    )
    compressor.fit(rows)

    print("=== Point Query ===")
    point = compressor.point_query(100)
    pprint(point)

    print("\n=== Range Query [100, 105] ===")
    result = compressor.range_query(100, 105)
    for row in result:
        pprint(row)

    print("\n=== Insert ===")
    new_row = {
        "id": 3001,
        "name": "Ivy",
        "city": "Shenzhen",
        "category": "E",
        "price": 99.99,
        "quantity": 7,
    }
    compressor.insert(new_row)
    pprint(compressor.point_query(3001))

    print("\n=== Update ===")
    updated_row = {
        "id": 3001,
        "name": "Ivy",
        "city": "Beijing",
        "category": "E",
        "price": 120.5,
        "quantity": 9,
    }
    compressor.update(3001, updated_row)
    pprint(compressor.point_query(3001))

    print("\n=== Delete ===")
    deleted = compressor.delete(3001)
    print("deleted:", deleted)
    print("after delete:", compressor.point_query(3001))

    report = compressor.estimate_memory(rows)
    print("\n=== Compression Statistics ===")
    print("original_bytes:", report.original_bytes)
    print("compressed_bytes:", report.compressed_bytes)
    print("compression_ratio:", round(report.compression_ratio, 4))


if __name__ == "__main__":
    main()
