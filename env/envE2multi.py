import gym
import numpy as np

class FinalTradingEnv(gym.Env):
    def __init__(self, models, base_env, render_mode=None):
        """
        최종 행동 결정을 위한 환경.
        Args:
            models (list): 학습된 PPO 모델 리스트 (5분, 15분, 1시간, 4시간 모델).
            base_env (gym.Env): 기본 환경 (주 데이터와 보상 체계 포함).
            render_mode (str): 렌더링 모드.
        """
        super(FinalTradingEnv, self).__init__()
        self.models = models  # 개별 시간 스케일 모델 (5분, 15분, 1시간, 4시간)
        self.base_env = base_env  # 보상과 실제 트레이딩 데이터를 다루는 기본 환경
        self.render_mode = render_mode

        # 관찰 공간: 각 모델의 정책 확률 (4개 모델 * 3개 액션 = 12개)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(models) * 3,), dtype=np.float32
        )

        # 액션 공간: 최종 행동 [-1, 0, 1]
        # self.action_space = gym.spaces.Discrete(3)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.base_env.reset()  # 기본 환경 초기화
        self.current_obs = self.base_env._get_observation()
        return self._get_combined_observation(), {}

    def _get_combined_observation(self):
        """각 모델의 정책 확률을 결합하여 반환"""
        policy_outputs = []
        for model in self.models:
            probs, _ = model.policy.forward(self.current_obs)
            policy_outputs.extend(probs.detach().cpu().numpy())
        return np.array(policy_outputs, dtype=np.float32)

    def step(self, action):
        """
        최종 행동을 기본 환경에 적용하고 보상 반환.
        Args:
            action (int): 최종 결정된 행동 (0: 유지, 1: 롱, 2: 숏).
        """
        # 기본 환경 액션 매핑
        # mapped_action = action - 1  # [-1, 0, 1]로 변환
        action = np.clip(action, -1, 1).item()
        if action > 0.5:
            action = 1  # 롱
        elif action < -0.5:
            action = -1  # 숏
        else:
            action = 0  # 유지
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        self.current_obs = obs  # 최신 관찰값 갱신
        return self._get_combined_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.base_env.render()
