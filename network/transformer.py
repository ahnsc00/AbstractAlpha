import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import math

# 사용자 정의 Transformer 기반 Features Extractor
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, seq_length: int = 10):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)
        self._features_dim = features_dim
        self.seq_length = seq_length  # 시퀀스 길이 추가

        # Embedding layer
        self.embedding = nn.Linear(observation_space.shape[0], features_dim)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(
            th.zeros(1, seq_length, features_dim), requires_grad=False
        )

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=features_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (batch_size, feature_dim)
        batch_size = observations.shape[0]

        # 시퀀스 차원 추가
        observations = observations.unsqueeze(1).repeat(1, self.seq_length, 1)  # (batch_size, seq_length, feature_dim)

        x = self.embedding(observations)
        x += self.positional_encoding.to(x.device)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 시퀀스 차원 평균
        return self.fc(x)

# 사용자 정의 정책
class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # use_sde 제거
        kwargs.pop("use_sde", None)
        super(TransformerPolicy, self).__init__(*args, **kwargs,
                                                features_extractor_class=TransformerFeatureExtractor,
                                                features_extractor_kwargs=dict(features_dim=128))