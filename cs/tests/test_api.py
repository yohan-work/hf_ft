from fastapi.testclient import TestClient

from cs_classifier.api import create_app
from cs_classifier.inference import PredictionResult


class FakePredictor:
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, text: str) -> PredictionResult:
        self.calls += 1
        return PredictionResult(
            label="payment",
            label_name="결제",
            confidence=0.91,
            scores=[{"label": "payment", "score": 0.91}, {"label": "delivery", "score": 0.09}],
        )


def test_health_and_predict_use_single_lifespan_predictor() -> None:
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_version": "v1", "device": "cpu"}
        response = client.post("/predict", json={"text": "결제했는데 주문이 없어요"})
    assert response.status_code == 200
    assert response.json()["label"] == "payment"
    assert response.json()["label_name"] == "결제"
    assert predictor.calls == 1


def test_predict_rejects_blank_and_too_long_text() -> None:
    with TestClient(create_app(FakePredictor)) as client:
        blank = client.post("/predict", json={"text": "  "})
        too_long = client.post("/predict", json={"text": "가" * 1001})
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"
    assert too_long.status_code == 422
