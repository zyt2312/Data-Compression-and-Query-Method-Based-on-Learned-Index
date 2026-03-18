from __future__ import annotations

import random


def generate_table(num_rows: int = 5000, seed: int = 42) -> list[dict[str, object]]:
    random.seed(seed)

    names = [
        "Alice",
        "Bob",
        "Carol",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
    ]
    cities = [
        "Shenzhen",
        "Guangzhou",
        "Shanghai",
        "Beijing",
        "Hangzhou",
        "Nanjing",
    ]
    categories = ["A", "B", "C", "D"]

    data: list[dict[str, object]] = []
    for idx in range(1, num_rows + 1):
        row = {
            "id": idx,
            "name": random.choice(names),
            "city": random.choice(cities),
            "category": random.choice(categories),
            "price": round(random.uniform(8, 300), 2),
            "quantity": random.randint(1, 50),
        }
        data.append(row)

    return data
