#%%
from gym_trading_env.downloader import download
import datetime
import pandas as pd
import gymnasium as gym
import gym_trading_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


download(exchange_names=["binance"],
         symbols=["BTC/USDT"],
         timeframe="4h",
         dir="../rltrader/rltrader/data",
         since=datetime.datetime(year=2018, month=1, day=1))


# 데이터 로드
df = pd.read_pickle("../rltrader/rltrader/data/binance-BTCUSDT-4h.pkl")

# 거래 환경 초기화
env = gym.make("TradingEnv",
               name="BTCUSD",
               df=df,
               positions=[-1, 0, 1],
               trading_fees=0.01/100,
               borrow_interest_rate=0.0003/100)

# 벡터화된 환경으로 포장
vec_env = DummyVecEnv([lambda: env])

# PPO 에이전트 초기화
model = PPO("MlpPolicy", vec_env, verbose=1)

# 에이전트 평가
obs = vec_env.reset()  # 벡터화된 환경에서는 obs만 반환됩니다.
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)  # 벡터화된 환경의 step 함수 사용
    vec_env.render()  # 환경 시각화



# 환경 초기화 및 평가
obs = vec_env.reset()
portfolio_values = []

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)

    # 포트폴리오 가치 저장 (Key 변경)
    portfolio_values.append(infos[0]['portfolio_valuation'])  # 첫 번째 환경의 포트폴리오 가치 저장

    if dones[0]:
        obs = vec_env.reset()

# 포트폴리오 가치 시각화
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values, label="Portfolio Value")
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.grid()
plt.show()
