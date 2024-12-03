import pandas as pd
def coin_load_data():
    # 데이터 로드
    btc_df = pd.read_pickle("./data/binance-BTCUSDT-4h.pkl")

    # 인덱스를 datetime으로 변환
    btc_df.index = pd.to_datetime(btc_df.index, errors='coerce')  # 'coerce'로 변환 불가능한 값을 NaT로 설정
    btc_df.index.rename('date', inplace=True)  # 인덱스 이름을 'date'로 설정

    def create_time_series_df(df):
        n = 50
        result = []

        dates = df.index  # 인덱스에서 직접 접근
        closes = df['close'].values

        # 슬라이딩 윈도우로 시계열 데이터 생성
        for i in range(len(df) - n + 1):
            date_slice = dates[i:i + n]
            close_slice = closes[i:i + n]
            if len(close_slice) == n:
                result.append([date_slice[0].strftime('%Y-%m-%d %H:%M')] + close_slice.tolist())  # 첫 날짜 저장

        # 데이터프레임 생성
        columns = ['date'] + [f'close_{i + 1}' for i in range(n)]
        time_series_df = pd.DataFrame(result, columns=columns)

        return time_series_df.astype(float, errors='ignore')

    # 시계열 데이터 생성
    df = create_time_series_df(btc_df)

    # 학습 데이터 준비
    coin_training_data = df.drop(columns=['date'])

    # 차트 데이터 준비
    coin_chart_data = btc_df.copy()

    # 인덱스를 문자열로 변환하면서 시간까지 포함
    coin_chart_data['date'] = coin_chart_data.index.strftime('%Y-%m-%d %H:%M')

    # 필요한 컬럼만 선택
    coin_chart_data = coin_chart_data[['date', 'open', 'high', 'low', 'close', 'volume']]

    # chartdata와 trainingdata 크기 조정
    # chartdata의 처음 50개 행 제거
    coin_chart_data = coin_chart_data.iloc[49:].reset_index(drop=True)

    # trainingdata의 마지막 50개 행 제거
    coin_training_data = coin_training_data.iloc[:-50].reset_index(drop=True)

    # 두 데이터프레임의 길이 동기화
    min_length = min(len(coin_chart_data), len(coin_training_data))
    coin_chart_data = coin_chart_data.iloc[:min_length]
    coin_training_data = coin_training_data.iloc[:min_length]

    # 시간 인덱스 맞추기
    coin_training_data['date'] = coin_chart_data['date'].values

    # chartdata와 trainingdata 병합
    combined_data = pd.concat([coin_chart_data, coin_training_data.drop(columns=['date'])], axis=1)
    combined_data.index = pd.to_datetime(combined_data.index, errors='coerce')
    combined_data = combined_data.set_index('date')
    combined_data.index = pd.to_datetime(combined_data.index)
    return coin_chart_data, coin_training_data, combined_data