import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym_trading_env
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from network.transformer import TransformerPolicy


# %%
def coin_load_data():
    # 데이터 로드
    btc_df = pd.read_pickle("../data/binance-BTCUSDT-15m.pkl")

    # 인덱스를 datetime으로 변환
    btc_df.index = pd.to_datetime(btc_df.index, errors='coerce')  # 'coerce'로 변환 불가능한 값을 NaT로 설정
    btc_df.index.rename('date', inplace=True)  # 인덱스 이름을 'date'로 설정

    def create_time_series_df(df):
        n = 100
        result = []

        dates = df.index  # 인덱스에서 직접 접근
        closes = df['close'].values

        # 첫 100개를 포함한 시퀀스 생성
        for i in range(len(df) - n + 1):
            date_slice = dates[i:i + n]
            close_slice = closes[i:i + n]
            if len(close_slice) == n:
                result.append([date_slice[-1].strftime('%Y-%m-%d %H:%M')] + close_slice.tolist())  # 마지막 날짜만 저장

        # 데이터프레임 생성
        columns = ['date'] + [f'close_{i + 1}' for i in range(n)]
        time_series_df = pd.DataFrame(result, columns=columns)

        # 데이터 부족 시 추가 처리 불필요
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

    # 데이터 타입 변환 및 병합
    coin_training_data['close_1'] = coin_training_data['close_1'].astype(float)
    coin_chart_data['close'] = coin_chart_data['close'].astype(float)

    # `close_1`과 `close`를 기준으로 병합
    merged_data = pd.merge_asof(
        coin_training_data.sort_values('close_1'),  # 병합 기준을 기준으로 정렬
        coin_chart_data.sort_values('close'),  # 병합 기준을 기준으로 정렬
        left_on='close_1',  # 학습 데이터의 close_1
        right_on='close',  # 차트 데이터의 close
        tolerance=0,  # 완전히 일치하는 값만 병합
        direction='forward'  # 순서대로 병합
    ).dropna()

    # 병합 후 date 컬럼을 datetime으로 변환
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    # 'date'를 기준으로 정렬
    merged_data = merged_data.sort_values(by='date')
    # date 컬럼을 인덱스로 설정
    merged_data.set_index('date', inplace=True)

    return coin_chart_data, coin_training_data, merged_data


# %%
chart_data, training_data, merged_data = coin_load_data()

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler를 행 단위로 적용
scaler = MinMaxScaler()

# 1번부터 100번 컬럼까지 각 행에 대해 정규화
merged_data.iloc[:, :100] = scaler.fit_transform(merged_data.iloc[:, :100].T).T
train, test = train_test_split(merged_data, test_size=0.2, random_state=42, shuffle=False)
# %%
# 거래 환경 초기화
env = gym.make("TradingEnv",
               name="BTCUSD",
               df=train,
               positions=[-1.0, 0, 1],
               trading_fees=0.02 / 100,
               borrow_interest_rate=0.0003 / 100)
# %%
# 환경 생성

# A2C 모델에 TransformerPolicy를 사용
# model = PPO(
#     policy=TransformerPolicy,
#     env=env,
#     learning_rate=2e-4,  # 안정적인 학습률 설정
#     clip_range=0.2,      # 클리핑 범위 설정
#     n_steps=2048,        # 업데이트 단계 수 증가
#     batch_size=256,      # 배치 크기 증가
#     ent_coef=0.01,       # 탐험 유도
#     vf_coef=0.5,         # 가치 함수 손실 가중치
#     max_grad_norm=0.5    # Gradient Clipping 설정
# )

model = PPO(
    policy=TransformerPolicy,
    env=env,
    verbose=1,
    device="cuda",
    learning_rate=2e-4
)

# %%
model.learn(total_timesteps=train.shape[0])
# %%


test_env = gym.make("TradingEnv",
                    name="BTCUSD",
                    df=test,
                    positions=[-1.0 + 0.25 * i for i in range(9)],
                    trading_fees=0.02 / 100,
                    borrow_interest_rate=0.0003 / 100)

# 환경 생성
# %%
# 환경 초기화 및 평가
obs, info = test_env.reset()
portfolio_values = []
i = 0
done, truncated = False, False
while not done and not truncated:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = test_env.step(action)
    i = i + 1
    print(i)
test_env.unwrapped.save_for_render(dir="../render_logs")  # 학습 후 에피소드 결과 저장
