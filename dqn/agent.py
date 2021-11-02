from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNet(nn.Module):
    def __init__(self, env_name):
        nn.Module.__init__(self)

        # CartPole: 2 linear layers with ReLU
        if env_name.startswith("CartPole"):
            num_inputs = 4
            num_outputs = 2
            layer_dims = [50, 25]
            self.layers = [nn.Linear(num_inputs, layer_dims[0]), nn.ReLU()]
            for i in range(len(layer_dims) - 1):
                self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layer_dims[-1], num_outputs))
            self.layers = nn.ModuleList(self.layers)

        # Breakout: 3 conv layers with ReLU
        elif env_name.startswith("Breakout"):
            self.layers = None  # TODO

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        raise NotImplementedError()


class RandomAgent(Agent):
    def __init__(self, action_space):
        Agent.__init__(self, action_space)

    def get_action(self, state):
        return self.action_space.sample()


class DQNAgent:
    def __init__(self, env_name, learning_rate, momentum, discount_factor):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_net = DQNet(env_name)
        self.q_target = deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate, momentum=momentum)

    def get_action(self, state):
        rewards = self.q_net(state)
        return torch.argmax(rewards).item()

    def get_q_target_estimate(self, exp_batch):
        state_next_batch = torch.cat([exp.state_next for exp in exp_batch])
        reward_batch = torch.tensor([exp.reward for exp in exp_batch])
        not_done_batch = torch.tensor([not exp.done for exp in exp_batch])
        q_batch = self.q_target(state_next_batch)
        return reward_batch + self.discount_factor * not_done_batch * q_batch.amax(axis=1)

    def get_q_value_estimate(self, exp_batch):
        state_batch = torch.cat([exp.state for exp in exp_batch])
        action_batch = torch.tensor([exp.action for exp in exp_batch])
        values = self.q_net(state_batch)
        return values.masked_select(F.one_hot(action_batch).bool())

    def reset_target(self):
        self.q_target = deepcopy(self.q_net)
