from pathlib import Path

import yaml


def test_label_ids_are_contiguous_and_codes_unique() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/labels.yaml").read_text(encoding="utf-8"))
    labels = config["labels"]
    assert [item["id"] for item in labels] == list(range(7))
    assert len({item["code"] for item in labels}) == 7
    assert {item["code"]: item["id"] for item in labels}["payment"] == 0
    assert {item["code"]: item["id"] for item in labels}["product_general"] == 6
