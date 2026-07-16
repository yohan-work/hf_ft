from pathlib import Path

from cs_classifier.data import SplitConfig, leakage_report, load_rows, split_rows


def test_split_is_reproducible_balanced_and_leak_free() -> None:
    root = Path(__file__).parents[1]
    labels = {
        "payment", "cancellation_refund", "delivery", "exchange_return", "account", "technical_issue", "product_general"
    }
    rows = load_rows(root / "data/raw/synthetic_cs_v1.csv", labels)
    first = split_rows(rows, SplitConfig())
    second = split_rows(rows, SplitConfig())
    assert {name: [row["id"] for row in values] for name, values in first.items()} == {
        name: [row["id"] for row in values] for name, values in second.items()
    }
    assert {name: len(values) for name, values in first.items()} == {"train": 98, "validation": 21, "test": 21}
    assert leakage_report(first)["passed"]
