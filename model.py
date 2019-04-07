import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        self._memory_size = memory_size = 1024
        self.memory_type = "GRU"

        # Define image embedding
        # kernel_size = 5; stride = 1
        # self.image_conv = nn.Sequential(
        #     nn.Conv2d(4, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU()
        # )

        kernel_size = 5; stride = 2
        no_last_filters = 32
        no_input_channels = 1
        self.image_conv = nn.Sequential(
                nn.Conv2d(no_input_channels, 16, kernel_size=kernel_size, stride=stride),
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

        # Define memory
        if self.use_memory:
            if self.memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, memory_size)
            else:
                self.memory_rnn = nn.GRUCell(self.image_embedding_size, memory_size)

            self.image_embedding_size = memory_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.image_embedding_size, 1024),
                nn.Tanh(),
                nn.Linear(1024, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 1024),
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
                hidden = (memory[:, :self._memory_size], memory[:, self._memory_size:])
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

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
