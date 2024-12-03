# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from data.coin_load_data import coin_load_data
#
#
# class TradingEnv(gym.Env):
#     def __init__(self, price_data, render_mode=None):
#         super(TradingEnv, self).__init__()
#         self.price_data = price_data.reset_index(drop=True)  # 인덱스 초기화
#         self.current_step = 0
#         self.entry_price = None
#         self.position = 0  # 1: 롱, -1: 숏, 0: 포지션 없음
#         self.render_mode = render_mode  # 렌더링 모드 추가
#         self.cum_reward = 0
#
#         # 액션 공간: 1 (롱), -1 (숏), 0 (유지), 2 (청산)
#         self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
#         # 관찰 공간: 가격 데이터 전체를 반환하도록 설정
#         self.observation_space = spaces.Box(
#             low=0, high=np.inf, shape=(price_data.shape[1],), dtype=np.float32
#         )
#
#     def reset(self, seed=None, **kwargs):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.entry_price = None
#         self.position = 0
#         return self._get_observation(), {}
#
#     def _get_observation(self):
#         # 현재 스텝의 가격 데이터를 반환
#         return self.price_data.iloc[self.current_step].to_numpy(dtype=np.float32)
#
#     def step(self, action):
#         reward = 0
#         terminated = False  # 에피소드 종료 여부
#         action = int(np.clip(action, -1, 1))
#
#         if action == 1:  # 롱 포지션 진입
#             if self.position == -1:  # 숏 포지션 청산
#                 current_price = self.price_data.iloc[self.current_step, 0]
#                 reward = self.position * np.log(current_price / self.entry_price)  # 숏 청산 보상
#                 self.position = 0  # 포지션 없음으로 전환
#                 self.entry_price = None
#             elif self.position == 0:  # 포지션 없음에서 롱 포지션 진입
#                 self.position = 1
#                 self.entry_price = self.price_data.iloc[self.current_step, 0]
#         elif action == -1:  # 숏 포지션 진입
#             if self.position == 1:  # 롱 포지션 청산
#                 current_price = self.price_data.iloc[self.current_step, 0]
#                 reward = self.position * np.log(current_price / self.entry_price)  # 롱 청산 보상
#                 self.position = 0  # 포지션 없음으로 전환
#                 self.entry_price = None
#             elif self.position == 0:  # 포지션 없음에서 숏 포지션 진입
#                 self.position = -1
#                 self.entry_price = self.price_data.iloc[self.current_step, 0]
#
#         self.cum_reward = self.cum_reward+reward
#         # 다음 스텝으로 이동
#         self.current_step += 1
#
#         if self.current_step >= len(self.price_data):
#             self.current_step = self.current_step = -1
#             terminated = True  # 데이터 끝에 도달하면 종료
#
#         return self._get_observation(), reward, terminated, False, {}
#
#     def render(self):
#         if self.render_mode == "human":
#             print(
#                 f"Step: {self.current_step}, Price: {self.price_data.iloc[self.current_step - 1, 0]}, Position: {self.position}, cun_reward:{self.cum_reward}"
#             )
#
#
# # %%
# chart_data, training_data, merged_data = coin_load_data()
# train, test = train_test_split(merged_data, test_size=0.2, random_state=42, shuffle=False)
# # %%
# # 환경 생성 및 감시(Monitor)
# env = TradingEnv(train, render_mode="human")
# env = Monitor(env)  # 학습 메트릭 기록
# env = DummyVecEnv([lambda: env])  # 벡터화된 환경으로 변환
#
# # PPO 에이전트 생성 및 학습
# model = PPO("MlpPolicy", env, verbose=1, device="cuda")
# model.learn(train.shape[0])
#
# test_env = TradingEnv(test, render_mode="human")
# test_env = DummyVecEnv([lambda: test_env])  # 벡터화된 환경으로 변환
# # 학습된 에이전트 테스트
# obs = test_env.reset()
# done = False
# total_reward = 0
#
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, _ = test_env.step(action)
#     test_env.render()
#     total_reward += reward
#
# print(f"Total Reward: {total_reward}")
from pyecharts.charts import Line, Grid, Scatter
from pyecharts import options as opts
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from sklearn.model_selection import train_test_split
from data.coin_load_data import coin_load_data


class TradingEnv(gym.Env):
    def __init__(self, price_data, render_mode=None):
        super(TradingEnv, self).__init__()
        self.price_data = price_data.reset_index(drop=True)  # 인덱스 초기화
        self.current_step = 0
        self.entry_price = None
        self.position = 0  # 1: 롱, -1: 숏, 0: 포지션 없음
        self.render_mode = render_mode  # 렌더링 모드 추가
        self.cum_reward = 0
        self.history = {"price": [], "position": [], "cum_reward": []}  # 그래프 데이터를 저장할 변수

        # 액션 공간: 1 (롱), -1 (숏), 0 (유지), 2 (청산)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # 관찰 공간: 가격 데이터 전체를 반환하도록 설정
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(price_data.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.entry_price = None
        self.position = 0
        self.cum_reward = 0
        # self.history = {"price": [], "position": [], "cum_reward": []}  # 기록 초기화
        return self._get_observation(), {}

    def _get_observation(self):
        # 현재 스텝의 가격 데이터를 반환
        return self.price_data.iloc[self.current_step].to_numpy(dtype=np.float32)

    def step(self, action):
        reward = 0
        terminated = False  # 에피소드 종료 여부
        action = int(np.clip(action, -1, 1).item())

        if action == 1:  # 롱 포지션 진입
            if self.position == -1:  # 숏 포지션 청산
                current_price = self.price_data.iloc[self.current_step, 0]
                reward = self.position * np.log(current_price / self.entry_price)  # 숏 청산 보상
                self.position = 0  # 포지션 없음으로 전환
                self.entry_price = None
            elif self.position == 0:  # 포지션 없음에서 롱 포지션 진입
                self.position = 1
                self.entry_price = self.price_data.iloc[self.current_step, 0]
        elif action == -1:  # 숏 포지션 진입
            if self.position == 1:  # 롱 포지션 청산
                current_price = self.price_data.iloc[self.current_step, 0]
                reward = self.position * np.log(current_price / self.entry_price)  # 롱 청산 보상
                self.position = 0  # 포지션 없음으로 전환
                self.entry_price = None
            elif self.position == 0:  # 포지션 없음에서 숏 포지션 진입
                self.position = -1
                self.entry_price = self.price_data.iloc[self.current_step, 0]

        self.cum_reward += reward

        # 기록 추가
        self.history["price"].append(self.price_data.iloc[self.current_step, 0])
        self.history["position"].append(self.position)
        self.history["cum_reward"].append(self.cum_reward)

        # 다음 스텝으로 이동
        self.current_step += 1
        if self.current_step >= len(self.price_data):
            terminated = True  # 데이터 끝에 도달하면 종료
            self.current_step=0

        return self._get_observation(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            print(
                f"Step: {self.current_step}, Price: {self.price_data.iloc[self.current_step - 1, 0]}, Position: {self.position}, cum_reward: {self.cum_reward}"
            )


# 환경 및 에이전트 설정
chart_data, training_data, merged_data = coin_load_data()
train, test = train_test_split(merged_data, test_size=0.2, random_state=42, shuffle=False)

env = TradingEnv(train, render_mode="human")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(train.shape[0])

test_env = TradingEnv(test, render_mode="human")
test_env = DummyVecEnv([lambda: test_env])

# 테스트
obs = test_env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = test_env.step(action)
    test_env.render()
    total_reward += reward
print(total_reward)

# Pyecharts 그래프 생성
history = test_env.envs[0].history

# 데이터 분리
price = history["price"]
position = history["position"]

# 포지션별로 가격 분리
long_positions = [price[i] if position[i] == 1 else None for i in range(len(price))]
short_positions = [price[i] if position[i] == -1 else None for i in range(len(price))]
neutral_positions = [price[i] if position[i] == 0 else None for i in range(len(price))]

line = (
    Line()
    .add_xaxis(list(range(len(price))))
    .add_yaxis("Price (Long)", long_positions, is_smooth=True, color="green")
    .add_yaxis("Price (Short)", short_positions, is_smooth=True, color="red")
    .add_yaxis("Price (Neutral)", neutral_positions, is_smooth=True, color="blue")
    .add_yaxis("Cumulative Reward", history["cum_reward"], is_smooth=True, color="orange")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Trading Results"),
        xaxis_opts=opts.AxisOpts(name="Steps"),
        yaxis_opts=opts.AxisOpts(name="Value"),
        tooltip_opts=opts.TooltipOpts(is_show=False)  # Disable tooltip
    )
)
line.render("trading_results_with_positions.html")