from cs_classifier.metrics import classification_metrics


def test_classification_metrics_reports_all_labels_including_missing_predictions() -> None:
    metrics = classification_metrics([0, 1], [0, 0], [0, 1], ["payment", "delivery"])
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.333333
    assert metrics["per_class"]["delivery"]["recall"] == 0.0
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]
