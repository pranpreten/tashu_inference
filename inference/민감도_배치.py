# -*- coding: utf-8 -*-
"""스테이션별 날씨(우천) 민감도 배치 계산 → station_meta.csv 생성

사용법:
    python 민감도_배치.py

원리 (재학습 없음 — 학습된 모델에 가정 질문을 두 번 던지는 what-if 시뮬레이션):
    같은 조건(스테이션·요일·시간·계절)에서 precip=0mm vs 5mm 두 번 예측
    → 대여 수요 하락률 = 이 스테이션이 비에 얼마나 약한가

산출:
    시나리오 격자 = 요일 7 × 시간 7~22시 × 계절 대표월 4개 (스테이션당 448개)
    하락률 = (Σ맑음예측 − Σ우천예측) / Σ맑음예측   ← 수요 가중 (한산한 시간대 노이즈 방지)

등급 기준 (방법1: 절대 고정 컷 — 분포 확인 결과 min 10.1% / max 37.8%):
    low    < 20%   (비가 와도 수요 유지 — 출퇴근 필수 이동 성격)
    medium < 30%
    high   ≥ 30%   (비 오면 수요 1/3 이상 증발 — 여가성 이용 성격)

재실행 시점: 모델 재학습 후 / 스테이션 추가·삭제 시. 그 외엔 이 CSV를 계속 재사용.
"""
import os

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'station_meta.csv')

STATIONS = ['ST0056', 'ST0065', 'ST0082', 'ST0107', 'ST0342', 'ST0627', 'ST0628',
            'ST0639', 'ST0641', 'ST0642', 'ST0685', 'ST0720', 'ST0934', 'ST1067',
            'ST1089', 'ST1302', 'ST1303', 'ST1330', 'ST1334']   # 학습 category 순서 — 변경 금지
X_COLS = ['station_id', 'hour', 'dow', 'is_weekend', 'month', 'is_holiday',
          'temp', 'humid', 'wind', 'precip', 'pm10', 'pm25', 'ozone']

RAIN_MM = 5.0                                    # 우천 시나리오 강수량
SEASONS = [(1, -2.0), (4, 15.0), (7, 28.0), (10, 16.0)]   # (월, 대표기온)
CUT_LOW, CUT_HIGH = 20.0, 30.0                   # 등급 절단점 (%)


def build_grid():
    """시나리오 격자: 19스테이션 × 요일7 × 시간 7~22 × 계절4 (precip=0 기준)"""
    rows = []
    for s in STATIONS:
        for dow in range(7):
            for hour in range(7, 23):
                for month, temp in SEASONS:
                    rows.append({'station_id': s, 'hour': hour, 'dow': dow,
                                 'is_weekend': int(dow >= 5), 'month': month, 'is_holiday': 0,
                                 'temp': temp, 'humid': 70.0, 'wind': 1.5, 'precip': 0.0,
                                 'pm10': 25.0, 'pm25': 14.0, 'ozone': 0.04})
    return pd.DataFrame(rows)


def main():
    model_rent = joblib.load(os.path.join(BASE, '..', 'models', 'model_rent_v1.pkl'))

    grid = build_grid()

    def predict(df):
        X = df[X_COLS].copy()
        X['station_id'] = pd.Categorical(X['station_id'], categories=STATIONS)
        return model_rent.predict(X).clip(min=0)

    dry = predict(grid)                          # 질문 1: 비 안 오면?
    wet_grid = grid.copy()
    wet_grid['precip'] = RAIN_MM
    wet = predict(wet_grid)                      # 질문 2: 비 5mm 오면? (그 외 조건 동일)

    r = pd.DataFrame({'station_id': grid['station_id'], 'dry': dry, 'wet': wet})
    agg = r.groupby('station_id').sum()
    meta = pd.DataFrame({
        'station_id': agg.index,
        'rain_drop_pct': ((agg['dry'] - agg['wet']) / agg['dry'] * 100).round(1).values,
    })
    meta['weather_sensitivity'] = pd.cut(
        meta['rain_drop_pct'], [-1, CUT_LOW, CUT_HIGH, 101], labels=['low', 'medium', 'high'])

    meta = meta.sort_values('rain_drop_pct', ascending=False)
    meta.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'저장: {OUT}')
    print(meta.to_string(index=False))


if __name__ == '__main__':
    main()
