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


def get_output_size(I, K, S, P=0):
    return (I - K + 2 * P) // S + 1


def get_multiple_output_size(shape, kernel_size, stride, no_last_filters):
    n, m = shape
    no_conv_layers = len(kernel_size)

    for i in range(no_conv_layers):
        n = get_output_size(n, kernel_size[i], stride[i])
        m = get_output_size(m, kernel_size[i], stride[i])
    return n * m * no_last_filters


class Actor(nn.Module):
    def __init__(self, state_size, history_size, use_memory=True, memory_type="LSTM"):
        super(Actor, self).__init__()
        self.use_memory = use_memory
        self.memory_type = memory_type

        # first fully connected
        self.base = nn.Sequential(
            nn.Linear(state_size + history_size, 128),
            nn.ReLU(inplace=True)
        )

        # add LSTM in case
        if self.use_memory:
            if self.memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(128, 128)
            else:
                self.memory_rnn = nn.GRUCell(128, 128)

        # steering head
        self.steer_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 181)
        )

        # acc & break head
        self.acc_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 201)
        )

    def forward(self, state, history, memory):
        # reshape
        x = state.reshape(state.shape[0], -1)
        history = history.reshape(history.shape[0], -1)

        # concatenate features with history
        x = torch.cat((x, history), dim=1)

        # pass concatenation through base
        x = self.base(x)

        # pass through LSTM in case
        if self.use_memory:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :128], memory[:, 128:])
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

        # compute steer distribution
        steer_dist = self.steer_head(embedding)
        steer_dist = Categorical(logits=F.log_softmax(steer_dist, dim=1))

        # compute acc distribution
        acc_dist = self.acc_head(embedding)
        acc_dist = Categorical(logits=F.log_softmax(acc_dist, dim=1))

        return (steer_dist, acc_dist), memory


class Critic(nn.Module):
    def __init__(self, state_size, history_size, params_size, use_memory, memory_type="LSTM"):
        super(Critic, self).__init__()
        self.history_size = history_size
        self.params_size = params_size

        self.use_memory = use_memory
        self.memory_type = memory_type

        # env parameters base
        self.params_base = nn.Sequential(
            nn.Linear(state_size + params_size, 128),
            nn.ReLU(inplace=True)
        )

        # action history base
        self.history_base = nn.Sequential(
            nn.Linear(state_size + history_size, 128),
            nn.ReLU(inplace=True)
        )

        # add LSTM in case
        if self.use_memory:
            if self.memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(128, 128)
            else:
                self.memory_rnn = nn.GRUCell(128, 128)

        #  head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, state, history, params, memory):
        # reshape
        state = state.reshape(state.shape[0], -1)
        history = history.reshape(history.shape[0], -1)
        params = params.reshape(params.shape[0], -1)

        # concatenate inputs
        x = torch.cat((state, params), dim=1)
        y = torch.cat((state, history), dim=1)

        # pass x through params base
        x = self.params_base(x)

        # pass y through history base
        y = self.history_base(y)

        # pass through LSTM in case
        if self.use_memory:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :128], memory[:, 128:])
                hidden = self.memory_rnn(y, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                hidden = memory
                hidden = self.memory_rnn(y, hidden)
                embedding = hidden
                memory = hidden
        else:
            embedding = y

        # concatenate x with embedding
        x = torch.cat((embedding, x), dim=1)

        # pass x trough head
        x = self.head(x)

        return x, memory


class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        self._memory_size = 2 * 128
        self.memory_type = "LSTM"

        obs_shape = (obs_space["image"][0], obs_space["image"][1])
        history_size = obs_space["action"]
        params_size = obs_space["params"]

        # feature extractor details
        self.kernel_size = [5, 5, 5]
        self.stride = [2, 2, 2]
        self.no_last_filters = 32
        self.no_input_channels = 4

        # image feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(self.no_input_channels, 16, kernel_size=self.kernel_size[0], stride=self.stride[0]),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=self.kernel_size[1], stride=self.stride[1]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.no_last_filters, kernel_size=self.kernel_size[2], stride=self.stride[2]),
            nn.BatchNorm2d(self.no_last_filters),
            nn.ReLU()
        )

        # compute state size after feature extractor
        self.state_size = get_multiple_output_size(
            obs_shape, kernel_size=self.kernel_size, stride=self.stride, no_last_filters=self.no_last_filters)

        # define actor
        self.actor = Actor(self.state_size, history_size,
                           use_memory=self.use_memory, memory_type=self.memory_type)

        # define critic
        self.critic = Critic(self.state_size, history_size, params_size,
                             use_memory=self.use_memory, memory_type=self.memory_type)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.memory_type == "LSTM":
            return 2 * self._memory_size
        else:
            return self._memory_size

    def forward(self, obs, memory):
        # split memory
        memory1, memory2 = memory[:, :self.memory_size//2], memory[:, self.memory_size//2:]

        # extract features
        state = self.feature(obs.image)

        # pass through actor
        (steer_dist, acc_dist), memory1 = self.actor(state, obs.action, memory1)

        # pass through ciritic
        value, memory2 = self.critic(state, obs.action, obs.params, memory2)
        value = value.squeeze(1)

        # concat memory
        memory = torch.cat((memory1, memory2), dim=1)

        return (steer_dist, acc_dist), value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
