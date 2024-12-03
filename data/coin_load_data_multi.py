import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def coin_load_data():
    # 데이터 로드
    btc_df = pd.read_pickle("./data/binance-BTCUSDT-5m.pkl")

    # 인덱스를 datetime으로 변환
    btc_df.index = pd.to_datetime(btc_df.index, errors='coerce')  # 'coerce'로 변환 불가능한 값을 NaT로 설정
    btc_df.index.rename('date', inplace=True)  # 인덱스 이름을 'date'로 설정

    def create_combined_data(df, window_size):
        result = []

        dates = df.index
        closes = df['close'].values

        # 슬라이딩 윈도우로 시계열 데이터 생성
        for i in range(len(df) - window_size + 1):
            date_slice = dates[i:i + window_size]
            close_slice = closes[i:i + window_size]
            if len(close_slice) == window_size:
                result.append([date_slice[0].strftime('%Y-%m-%d %H:%M')] + close_slice.tolist())

        # 데이터프레임 생성
        columns = ['date'] + [f'close_{i + 1}' for i in range(window_size)]
        time_series_df = pd.DataFrame(result, columns=columns)

        # 학습 데이터 준비
        training_data = time_series_df.drop(columns=['date'])

        # 차트 데이터 준비
        chart_data = df.copy()
        chart_data['date'] = chart_data.index.strftime('%Y-%m-%d %H:%M')
        chart_data = chart_data[['date', 'open', 'high', 'low', 'close', 'volume']]

        # 크기 조정
        chart_data = chart_data.iloc[window_size - 1:].reset_index(drop=True)
        training_data = training_data.iloc[:-window_size + 1].reset_index(drop=True)

        # 길이 동기화
        min_length = min(len(chart_data), len(training_data))
        chart_data = chart_data.iloc[:min_length]
        training_data = training_data.iloc[:min_length]

        # 시간 인덱스 맞추기
        training_data['date'] = chart_data['date'].values

        #정규화
        scaler = MinMaxScaler()
        training_data = pd.DataFrame(scaler.fit_transform(training_data.T).T, columns=training_data.columns)

        # 데이터 병합
        combined_data = pd.concat([chart_data, training_data.drop(columns=['date'])], axis=1)
        combined_data.index = pd.to_datetime(combined_data['date'], errors='coerce')
        combined_data = combined_data.set_index('date')
        return combined_data

    # 윈도우 크기 설정 (5m 데이터 기준)
    window_sizes = {
        '5m': 50,  # 50 * 5m = 250m = 약 4시간
        '15m': 150,  # 150 * 5m = 750m = 약 12시간
        '1h': 300,  # 300 * 5m = 1500m = 약 1일
        '4h': 600,  # 1200 * 5m = 6000m = 약 4일
    }

    # 각 간격의 combined_data 생성
    combined_data_5m = create_combined_data(btc_df, window_sizes['5m'])
    combined_data_15m = create_combined_data(btc_df, window_sizes['15m'])
    combined_data_1h = create_combined_data(btc_df, window_sizes['1h'])
    combined_data_4h = create_combined_data(btc_df, window_sizes['4h'])

    return combined_data_5m, combined_data_15m, combined_data_1h, combined_data_4h