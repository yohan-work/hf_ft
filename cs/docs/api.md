# 로컬 추론 API

저장된 `models/klue-roberta-cs-v1` 모델을 시작 시 한 번 로드하고, 요청마다 재로드하지 않는
FastAPI API다.

## 실행

```bash
cd cs
uv run uvicorn cs_classifier.api:app --host 127.0.0.1 --port 8000
```

`127.0.0.1`로만 바인딩하므로 로컬 컴퓨터 외부에 공개되지 않는다.

## 요청

```bash
curl --request POST http://127.0.0.1:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{"text":"결제는 완료됐는데 주문 내역이 보이지 않아요."}'
```

응답 예시:

```json
{
  "label": "payment",
  "label_name": "결제",
  "confidence": 0.179227,
  "scores": [
    {"label": "payment", "score": 0.179227}
  ],
  "model_version": "v1"
}
```

`confidence`는 이 인공 소규모 데이터 모델의 운영 확정 기준이 아니다. 특히 취소·환불 성능이
낮으므로 자동 처리에 사용하지 않는다.

## 상태 확인과 오류

```bash
curl http://127.0.0.1:8000/health
```

빈 문장이나 1,000자를 넘는 입력은 HTTP 422와 다음 형식으로 거부한다.

```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid prediction request",
    "details": []
  }
}
```

## 환경 변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `CS_MODEL_PATH` | `models/klue-roberta-cs-v1` | 저장 모델 경로 |
| `CS_LABELS_PATH` | `configs/labels.yaml` | 라벨 한글명 설정 경로 |
| `CS_MODEL_VERSION` | `v1` | 응답에 표시할 버전 |
| `CS_MAX_LENGTH` | `128` | tokenizer 최대 token 길이 |
| `CS_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` 중 선택 |

현재 환경에서는 MPS가 사용 불가하므로 `auto`는 CPU를 선택한다.
