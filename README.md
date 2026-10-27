# 대전 타슈 대여·반납 수요 예측

대전 공영자전거 타슈의 이용 이력과 날씨·대기질 자료로 대여소별 시간당 대여·반납 수요를 예측하는 프로젝트다. 데이터 탐색과 전처리, 모델 학습, 향후 시간대의 수요를 출력하는 CLI 추론 코드를 포함한다.

## 처리 흐름

`이용·환경 데이터 → 시간·대여소별 집계 → 시간순 분할 → LightGBM 학습 → 기상 예보·현재 대기질 기반 추론`

- 대여와 반납을 별도 모델로 학습
- 대여소·시간·요일·월·공휴일 및 기상·대기질 특성 사용
- 기상청 단기예보와 에어코리아 관측값 조회
- 다음 정시부터 모레 23시까지 시간별 예측
- 대여량에서 반납량을 뺀 순유출량 및 대여소 날씨 민감도 표시

## 파일 구성

| 경로 | 역할 |
|---|---|
| `eda/` | 이용·환경 데이터 탐색 |
| `preprocessing/` | 집계·결합·분할과 train·val·test 자료 |
| `training/모델학습.ipynb` | LightGBM 모델 학습·평가 |
| `training/모델학습_xgboost.ipynb` | XGBoost 실험 |
| `models/model_rent_v1.pkl`, `models/model_return_v1.pkl` | 대여·반납 모델 |
| `models/model_v1_meta.json` | 모델 메타정보 |
| `inference/tashu_predict.py` | 실시간 입력 구성과 CLI 추론 |
| `inference/station_meta.csv` | 대여소별 민감도 자료 |

## 데이터

저장소에 타슈 이용·대여소·날씨·대기질 CSV와 전처리 결과가 포함되어 있다. 집계 시 이용 레코드 수가 아니라 `qty` 합을 사용한다. 학습과 평가의 시간 구간을 구분하며, 자료가 없는 기간을 정상적인 0 수요로 해석하지 않는다.

추론에서는 외부 API가 필요하다. 본인의 공공데이터포털 인증키로 기상청 단기예보와 에어코리아 API 활용을 준비하고 환경변수 `SERVICE_KEY`를 설정한다. 저장소 코드의 기본 인증값에 의존하지 않고 자기 환경의 값을 사용한다.

## 실행 방법

저장소 루트의 Windows PowerShell에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy lightgbm scikit-learn joblib requests holidays
$env:SERVICE_KEY = "본인_API_인증키"
.\.venv\Scripts\python.exe inference/tashu_predict.py ST0065 --input
```

`--input`은 모델 입력 특성을 함께 표시한다. 사용 가능한 대여소 ID는 `tashu_predict.py`의 `STATIONS`를 확인한다. 모델 파일을 읽지 못하면 학습 당시 LightGBM·scikit-learn 버전과 현재 환경의 호환성을 점검한다.

재학습은 Jupyter 환경에서 전처리 노트북과 학습 노트북을 순서대로 실행한다. 학습 노트북은 `training/` 기준 상대 경로를 사용한다.

## 결과와 제한 사항

시간별 `rent`, `return`, `net(rent-return)`을 출력한다. 순유출이 양수이면 예측 대여가 반납보다 많다는 뜻이며 현재 자전거 재고가 부족하다고 확정하는 값은 아니다.

미래 대기질은 예보가 아니라 현재 관측값을 모든 시점에 적용하는 근사다. 기상 예보의 먼 시각이 비면 직전 예보를 재사용한다. 따라서 예측 오차에는 모델 오차와 외부 입력의 근사가 함께 반영된다. 대여소나 서비스 지역을 확장하려면 별도 데이터와 재검증이 필요하다.

---

[GitHub 저장소](https://github.com/pranpreten/tashu_inference) · [문서 작성 기준 코드](https://github.com/pranpreten/tashu_inference/tree/857ce33b2d95737896cbbd4c116a19d1a5557015)
