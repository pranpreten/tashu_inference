# -*- coding: utf-8 -*-
"""타슈 실시간 수요예측 (현재시각 ~ +2시간, 총 3시간)

사용법:
    python tashu_predict.py ST0065

    예) 지금이 13:27이면 → 13시, 14시, 15시의 대여/반납 예측 출력

API 키 (data.go.kr 공공데이터포털 인증키 1개로 둘 다 됨. 각각 '활용신청'만 필요):
    1. 기상청_단기예보 ((구)_동네예보) 조회서비스   → temp/humid/wind/precip
    2. 한국환경공단_에어코리아_대기오염정보          → pm10/pm25/ozone
    발급받은 일반 인증키(Decoding)를 환경변수 SERVICE_KEY 또는 아래 상수에 입력.
"""
import os
import re
import sys
from datetime import datetime, timedelta

import joblib
import pandas as pd
import requests
import holidays

SERVICE_KEY = os.environ.get('SERVICE_KEY', '57b02d67aab97456724a40b4d0f96782b644a26f78308ef9063613898b4d3d18')
NX, NY = 68, 100          # 대전역 일대 기상청 격자 (격자변환표로 검증 필요)
BASE = os.path.dirname(os.path.abspath(__file__))


def _get_json(url, params):
    """data.go.kr 호출 — 인증 실패 시 XML로 응답하므로 JSON 파싱 전에 에러를 잡아 보여줌"""
    r = requests.get(url, params=params, timeout=10)
    try:
        return r.json()
    except ValueError:
        body = r.text[:300]
        hint = ''
        if 'SERVICE_KEY_IS_NOT_REGISTERED' in body:
            hint = ('\n→ 키가 아직 서버에 미등록. 활용신청 직후라면 30분~1시간 뒤 재시도.'
                    '\n→ 마이페이지 > 활용신청 목록에서 "일반 인증키(Decoding)"를 복사했는지 확인.')
        elif 'LIMITED_NUMBER_OF_SERVICE_REQUESTS' in body:
            hint = '\n→ 일일 트래픽(10,000) 초과.'
        elif 'SERVICE ERROR' in body or 'APPLICATION_ERROR' in body:
            hint = '\n→ 해당 API의 활용신청 승인 여부를 마이페이지에서 확인.'
        raise SystemExit(f'[API 에러] {url.rsplit("/", 1)[-1]} 가 JSON이 아닌 응답을 반환:\n{body}{hint}')

STATIONS = ['ST0056', 'ST0065', 'ST0082', 'ST0107', 'ST0342', 'ST0627', 'ST0628',
            'ST0639', 'ST0641', 'ST0642', 'ST0685', 'ST0720', 'ST0934', 'ST1067',
            'ST1089', 'ST1302', 'ST1303', 'ST1330', 'ST1334']   # 학습 category 순서 — 변경 금지
X_COLS = ['station_id', 'hour', 'dow', 'is_weekend', 'month', 'is_holiday',
          'temp', 'humid', 'wind', 'precip', 'pm10', 'pm25', 'ozone']
KR_HOLIDAYS = holidays.KR(years=range(2025, 2031))
EXTRA_HOLIDAYS = set()    # 임시공휴일 발표되면 {'2026-xx-xx'} 형태로 추가


_BASE_TIMES = [23, 20, 17, 14, 11, 8, 5, 2]   # 단기예보 발표시각 (1일 8회)


def _forecast_base(now):
    """가장 최근에 제공된 단기예보 발표시각 (base_date, base_time). 발표 후 ~10분 제공 → 여유 15분."""
    t = now - timedelta(minutes=15)
    for h in _BASE_TIMES:
        if t.hour >= h:
            return t.strftime('%Y%m%d'), f'{h:02d}00'
    y = t - timedelta(days=1)                  # 자정~02시대: 전날 23시 발표분 사용
    return y.strftime('%Y%m%d'), '2300'


def fetch_forecast():
    """기상청 단기예보(getVilageFcst, +3일) → {(YYYYMMDD, HH00): {temp,humid,wind,precip}}

    3일 예측이라 초단기(+6h) 대신 단기예보 사용. 항목명이 초단기와 다름: T1H→TMP, RN1→PCP.
    """
    base_date, base_time = _forecast_base(datetime.now())
    data = _get_json(
        'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst',
        params={'serviceKey': SERVICE_KEY, 'dataType': 'JSON',
                'base_date': base_date, 'base_time': base_time,
                'nx': NX, 'ny': NY, 'numOfRows': 1000, 'pageNo': 1})   # 3일치 전 항목 ~900건
    header = data['response']['header']
    if header['resultCode'] != '00':
        raise SystemExit(f"[기상청 API 에러] {header['resultCode']}: {header['resultMsg']}")
    items = data['response']['body']['items']['item']

    def pcp(v):  # PCP: '강수없음' | '1mm 미만' | '1.0mm' | '30.0~50.0mm' | '50.0mm 이상'
        if v in (None, '강수없음', '-', ''): return 0.0
        s = str(v)
        if '미만' in s: return 0.5
        nums = re.findall(r'[\d.]+', s)
        return float(nums[-1]) if nums else 0.0

    fc = {}
    for it in items:
        fc.setdefault((it['fcstDate'], it['fcstTime']), {})[it['category']] = it['fcstValue']
    return {k: {'temp': float(v['TMP']), 'humid': float(v['REH']),
                'wind': float(v['WSD']), 'precip': pcp(v.get('PCP'))}
            for k, v in fc.items() if 'TMP' in v}


def fetch_air():
    """에어코리아 성남동1 최신 관측값 → {pm10, pm25, ozone} (미래 수치예보 없음 → 현재값 근사)"""
    data = _get_json(
        'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty',
        params={'serviceKey': SERVICE_KEY, 'returnType': 'json', 'stationName': '성남동1',
                'dataTerm': 'DAILY', 'numOfRows': 24, 'pageNo': 1, 'ver': '1.3'})
    items = data['response']['body']['items']

    def latest(key):  # 통신장애('-') 건너뛰고 최신 유효값
        return next((float(it[key]) for it in items if it.get(key) not in (None, '-', '')), None)
    return {'pm10': latest('pm10Value'), 'pm25': latest('pm25Value'), 'ozone': latest('o3Value')}


def predict_horizon(station_id, forecast=None, air=None, now=None, show_input=False):
    """station_id의 다음 정시 ~ 내일모레(D+2) 23시 예측 → DataFrame[time, rent, return]

    예) 지금이 09:xx면 → 오늘 10시부터 오늘/내일/내일모레 23시까지 매시각.
    """
    if station_id not in STATIONS:
        raise SystemExit(f'모르는 스테이션: {station_id}\n가능: {", ".join(STATIONS)}')
    forecast = forecast if forecast is not None else fetch_forecast()
    air = air if air is not None else fetch_air()
    now = now or datetime.now()

    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)   # 다음 정시
    end   = now.replace(hour=23, minute=0, second=0, microsecond=0) + timedelta(days=2)  # 내일모레 23시
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += timedelta(hours=1)

    rows, last_w = [], None
    for t in times:
        w = forecast.get((t.strftime('%Y%m%d'), f'{t.hour:02d}00'))
        if w is None:                      # 예보 커버리지 밖(먼 시각)이면 직전 값으로 채움
            if last_w is None:
                raise SystemExit(f'{t:%Y-%m-%d %H}시 예보 없음 (base_time 직후엔 드물게 발생 — 잠시 후 재시도)')
            w = last_w
        last_w = w
        rows.append({'station_id': station_id, 'hour': t.hour,
                     'dow': t.weekday(), 'is_weekend': int(t.weekday() >= 5), 'month': t.month,
                     'is_holiday': int(t.date() in KR_HOLIDAYS or t.strftime('%Y-%m-%d') in EXTRA_HOLIDAYS),
                     **w, **air})

    X = pd.DataFrame(rows)[X_COLS]
    X['station_id'] = pd.Categorical(X['station_id'], categories=STATIONS)  # 학습과 동일 인코딩

    if show_input:                                   # --input: 모델 인풋 확인용
        dbg = X.copy()
        dbg.insert(0, 'time', [t.strftime('%m-%d %H시') for t in times])
        print(f'\n=== 모델 인풋 (총 {len(dbg)}행 × 피처 13개) — 앞 8행 ===')
        print(dbg.head(8).to_string(index=False))
        print('  · station_id~is_holiday: 입력값+달력 파생 | temp~precip: 기상청 시각별 예보 | pm10~ozone: 에어코리아 현재값(전 행 동일)')

    model_rent = joblib.load(os.path.join(BASE, '..', 'models', 'model_rent_v1.pkl'))
    model_return = joblib.load(os.path.join(BASE, '..', 'models', 'model_return_v1.pkl'))

    return pd.DataFrame({
        'time':   [t.strftime('%m-%d %H시') for t in times],
        'rent':   model_rent.predict(X).round(2),
        'return': model_return.predict(X).round(2),
    })


def load_sensitivity(station_id):
    """배치 산출물 station_meta.csv에서 우천 민감도 조회 (없으면 None — 민감도_배치.py 실행 필요)

    민감도는 스테이션 고정 속성이라 행마다 넣지 않고 헤더에 1회만 표시.
    """
    path = os.path.join(BASE, 'station_meta.csv')
    if not os.path.exists(path):
        return None
    meta = pd.read_csv(path)
    row = meta[meta['station_id'] == station_id]
    if row.empty:
        return None
    return row.iloc[0]['weather_sensitivity'], row.iloc[0]['rain_drop_pct']


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--input']
    if len(args) != 1:
        raise SystemExit(f'사용법: python {os.path.basename(__file__)} <station_id> [--input]  예) ST0065')
    result = predict_horizon(args[0], show_input='--input' in sys.argv)
    result['net(rent-return)'] = (result['rent'] - result['return']).round(2)  # +면 자전거 유출

    sens = load_sensitivity(sys.argv[1])
    sens_txt = f' | 날씨민감도: {sens[0]} (우천 시 수요 약 {sens[1]:.0f}% 하락)' if sens else ''
    print(f'\n[{sys.argv[1]}] 다음 정시 ~ 내일모레 23시 수요 예측{sens_txt}')
    print(result.to_string(index=False))
