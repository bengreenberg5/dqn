import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNet(nn.Module):
    def __init__(self, env_name, history_length):
        nn.Module.__init__(self)

        # CartPole: 2 linear layers with ReLU
        if env_name.startswith("CartPole"):
            num_inputs = 4
            num_outputs = 2
            self.layers = [
                nn.Linear(num_inputs, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, num_outputs)
            ]

        # Breakout: 3 conv layers with ReLU
        elif env_name in ["Breakout-v4"]:
            num_outputs = 4
            self.layers = [
                nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 8, 8), stride=(1, 4, 4), padding="valid"),
                nn.ReLU(),
                nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding="valid"),
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding="valid"),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=64*history_length*7*7, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=num_outputs)
            ]

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DQNAgent:
    def __init__(self, env_name, history_length, learning_rate, momentum, discount_factor):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_net = DQNet(env_name, history_length)
        self.q_target = deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(), lr=learning_rate, momentum=momentum
        )

    def save_networks(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save(self.q_net, f"{dirname}/q_net.pt")
        torch.save(self.q_target, f"{dirname}/q_target.pt")

    def get_action(self, state):
        rewards = self.q_net(state)
        return torch.argmax(rewards).item()

    def get_q_target_estimate(self, exp_batch):
        state_next_batch = torch.cat([exp.state_next for exp in exp_batch], dim=0)
        reward_batch = torch.tensor([exp.reward for exp in exp_batch])
        not_done_batch = torch.tensor([not exp.done for exp in exp_batch])
        q_batch = self.q_target(state_next_batch)
        return reward_batch + self.discount_factor * not_done_batch * q_batch.amax(
            axis=1
        )

    def get_q_value_estimate(self, exp_batch):
        state_batch = torch.cat([exp.state for exp in exp_batch], dim=0)
        action_batch = torch.tensor([exp.action for exp in exp_batch])
        values = self.q_net(state_batch)
        return values.masked_select(F.one_hot(action_batch, num_classes=4).bool())

    def reset_target(self):
        self.q_target = deepcopy(self.q_net)
