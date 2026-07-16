import numpy as np

from cs_classifier.training import make_compute_metrics


class Prediction:
    predictions = np.array([[3.0, 1.0], [0.1, 0.9]])
    label_ids = np.array([0, 1])


def test_trainer_metrics_use_macro_f1_and_label_order() -> None:
    metrics = make_compute_metrics(["payment", "delivery"])(Prediction())
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["payment"]["support"] == 1
