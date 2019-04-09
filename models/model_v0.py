# parts from https://github.com/lcswillems/torch-rl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym

from models.utils import initialize_parameters


class ModelV0(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, cfg, obs_space, action_space, use_memory=False, no_stacked_frames=4):
        super().__init__()

        # CFG Information
        self.memory_type = memory_type = cfg.memory_type
        self._memory_size = memory_size = getattr(cfg, "memory_size", 1024)

        # Decide which components are enabled
        self.use_memory = use_memory

        # Define image embedding
        kernel_size = 5; stride = 2
        no_last_filters = 32
        self.image_conv = nn.Sequential(
                nn.Conv2d(no_stacked_frames, 16, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, no_last_filters, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(no_last_filters),
                nn.ReLU()
        )

        n = obs_space["image"][0]
        m = obs_space["image"][1]

        def get_output_size(I, K, S, P=0):
            return (I - K + 2 * P) // S + 1

        no_conv_layers = 3
        for _ in range(no_conv_layers):
            n = get_output_size(n, kernel_size, stride)
            m = get_output_size(m, kernel_size, stride)
        

        self.image_embedding_size = n * m * no_last_filters
        crt_size = self.image_embedding_size

        # Define memory
        if self.use_memory:
            if memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(crt_size, memory_size)
            else:
                self.memory_rnn = nn.GRUCell(crt_size, memory_size)

            crt_size = memory_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(crt_size, 1024),
                nn.Tanh(),
                nn.Linear(1024, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(crt_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.memory_type == "LSTM":
            return 2 * self._memory_size
        else:
            return self._memory_size

    def forward(self, obs, memory):
        x = obs.image
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                hidden = memory
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden
                memory = hidden
        else:
            embedding = x

        x = self.actor(embedding)
        steer_dist = Categorical(logits=F.log_softmax(x[:, :181], dim=1))
        acc_dist = Categorical(logits=F.log_softmax(x[:, 181:], dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return (steer_dist, acc_dist), value, memory

