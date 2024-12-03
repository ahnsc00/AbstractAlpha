import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, merged_data, render_mode=None):
        super(TradingEnv, self).__init__()
        self.training = merged_data.iloc[:, 5:]
        self.price_data = merged_data.iloc[:, :5].reset_index(drop=True)  # 인덱스 초기화
        self.current_step = 0
        self.entry_price = None
        self.position = 0  # 1: 롱, -1: 숏, 0: 포지션 없음
        self.render_mode = render_mode  # 렌더링 모드 추가
        self.cum_reward = 0
        self.history = {"price": [], "position": [], "cum_reward": []}  # 그래프 데이터를 저장할 변수

        # 액션 공간: 1 (롱), -1 (숏), 0 (유지)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        # 관찰 공간: 가격 데이터 전체를 반환하도록 설정

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.training.shape[1],), dtype=np.float32
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
        return self.training.iloc[self.current_step].to_numpy(dtype=np.float32)

    def step(self, action):
        reward = 0
        terminated = False  # 에피소드 종료 여부
        action = action - 1
        # action = np.clip(action, -1, 1).item()
        # if action > 0.5:
        #     action = 1  # 롱
        # elif action < -0.5:
        #     action = -1  # 숏
        # else:
        #     action = 0  # 유지

        if action == 1:  # 롱 포지션 진입 또는 롱으로 전환
            if self.position == -1:  # 숏 포지션 청산 후 롱으로 전환
                current_price = self.price_data.iloc[self.current_step, 0]
                profit = (self.entry_price - current_price) / self.entry_price  # 숏 청산 수익률
                # reward = profit - 0.0002  # 수수료 차감
                reward = profit
                self.position = 1  # 롱 포지션으로 전환
                self.entry_price = current_price  # 새로운 진입 가격
                # reward -= 0.0002  # 롱 진입 수수료 추가
            elif self.position == 0:  # 포지션 없음에서 롱 진입
                self.position = 1
                self.entry_price = self.price_data.iloc[self.current_step, 0]
                # reward = -0.0002  # 진입 수수료만 반영

        elif action == -1:  # 숏 포지션 진입 또는 숏으로 전환
            if self.position == 1:  # 롱 포지션 청산 후 숏으로 전환
                current_price = self.price_data.iloc[self.current_step, 0]
                profit = (current_price - self.entry_price) / self.entry_price  # 롱 청산 수익률
                reward = profit
                # reward = profit - 0.0002  # 수수료 차감
                self.position = -1  # 숏 포지션으로 전환
                self.entry_price = current_price  # 새로운 진입 가격
                # reward -= 0.0002  # 숏 진입 수수료 추가
            elif self.position == 0:  # 포지션 없음에서 숏 진입
                self.position = -1
                self.entry_price = self.price_data.iloc[self.current_step, 0]
                # reward = -0.0002  # 진입 수수료만 반영

        elif action == 0:  # 포지션 청산
            if self.position == 1:  # 롱 포지션 청산
                current_price = self.price_data.iloc[self.current_step, 0]
                reward = (current_price - self.entry_price) / self.entry_price   # 롱 청산 수익률 + 수수료 차감
                # reward = reward- 0.0002
                self.position = 0  # 포지션 없음으로 초기화
                self.entry_price = None
            elif self.position == -1:  # 숏 포지션 청산
                current_price = self.price_data.iloc[self.current_step, 0]
                reward = (self.entry_price - current_price) / self.entry_price  # 숏 청산 수익률 + 수수료 차감
                # reward = reward - 0.0002
                self.position = 0  # 포지션 없음으로 초기화
                self.entry_price = None
            else:
                reward = 0  # 이미 포지션 없음

        self.cum_reward += reward

        # 기록 추가
        self.history["price"].append(self.price_data.iloc[self.current_step, 0])
        self.history["position"].append(self.position)
        self.history["cum_reward"].append(self.cum_reward)
        observation = self._get_observation()

        # 다음 스텝으로 이동
        self.current_step += 1
        if self.current_step >= len(self.price_data):
            terminated = True  # 데이터 끝에 도달하면 종료
            self.current_step = 0

        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            print(
                f"Step: {self.current_step}, Price: {self.price_data.iloc[self.current_step - 1, 0]}, Position: {self.position}, cum_reward: {self.cum_reward}"
            )