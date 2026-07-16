# Korean CS Classifier Lab

Hugging Face 기반 한국어 고객 문의 분류를 단계적으로 학습하는 프로젝트다.

현재 구현 범위는 Phase 4 데이터 파이프라인이다. 실행 환경을 준비한 뒤 다음 명령으로
원본 인공 데이터를 재현 가능한 split과 토큰화 통계로 변환한다.

```bash
uv run python scripts/prepare_data.py
uv run pytest
```

모델 학습, Hub 업로드, API는 이후 Phase에서 추가한다.
