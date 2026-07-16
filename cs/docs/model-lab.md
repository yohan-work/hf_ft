# Model Lab

로컬 FastAPI 서버의 `/`에서 열리는 최소 실험 화면이다. 운영용 CS 대시보드가 아니라 모델의
예측과 평가 근거를 한곳에서 확인하는 도구다.

## 실행

```bash
cd cs
uv run uvicorn cs_classifier.api:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 연다.

## 화면 기능

- 문의 입력과 `/predict` 호출
- 7개 라벨 전체 확률과 최고 예측 표시
- 모델 버전·실행 장치 표시
- 고정 Test의 baseline/Transformer Macro F1 비교
- Transformer Confusion Matrix와 클래스별 F1
- Transformer 오분류 6개와 confidence

평가 화면은 `artifacts/metrics/final-test-metrics.json`을 읽는다. 이 산출물이 없으면 평가 영역은
비어 있으며, 먼저 Phase 7의 `scripts/evaluate.py`를 실행해야 한다.
