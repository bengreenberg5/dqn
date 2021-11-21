from copy import deepcopy
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNet(nn.Module):
    def __init__(self, env_name, history_length):
        nn.Module.__init__(self)
        self.num_outputs = gym.make(env_name).action_space.n

        # CartPole: 2 linear layers with ReLU
        if env_name.startswith("CartPole"):
            num_inputs = 4
            self.layers = [
                nn.Linear(num_inputs, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, self.num_outputs),
            ]

        # Breakout: 3 conv layers with ReLU
        elif env_name.startswith("Breakout"):
            self.layers = [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(1, 8, 8),
                    stride=(1, 4, 4),
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    padding="valid",
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=64 * history_length * 7 * 7, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=self.num_outputs),
            ]

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DQNAgent:
    def __init__(
        self,
        env_name,
        history_length=4,
        learning_rate=1e-4,
        momentum=0.95,
        discount_factor=0.99,
    ):
        self.env_name = env_name
        self.history_length = history_length
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net = DQNet(env_name, history_length).to(self.device)
        self.q_target = deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(), lr=learning_rate, momentum=momentum
        )

    def save_networks(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save(self.q_net.state_dict(), f"{dirname}/q_net.pt")
        torch.save(self.q_target.state_dict(), f"{dirname}/q_target.pt")

    def load_networks(self, dirname, checkpoint):
        assert os.path.exists(dirname), f"directory {dirname} does not exist"
        checkpoint = str(checkpoint).zfill(7)
        self.q_net.load_state_dict(torch.load(f"{dirname}/{checkpoint}/q_net.pt"))
        self.q_target.load_state_dict(torch.load(f"{dirname}/{checkpoint}/q_target.pt"))

    def get_action(self, state):
        rewards = self.q_net(state.float().to(self.device))
        return torch.argmax(rewards).item()

    def get_q_target_estimate(self, exp_batch):
        state_next_batch = torch.cat(
            [exp.state_next.float() for exp in exp_batch], dim=0
        ).to(self.device)
        reward_batch = torch.tensor([exp.reward for exp in exp_batch]).to(self.device)
        not_done_batch = torch.tensor([not exp.done for exp in exp_batch]).to(
            self.device
        )
        q_batch = self.q_target(state_next_batch.to(self.device))
        return reward_batch + self.discount_factor * not_done_batch * q_batch.amax(
            axis=1
        )

    def get_q_value_estimate(self, exp_batch):
        state_batch = torch.cat([exp.state.float() for exp in exp_batch], dim=0).to(
            self.device
        )
        action_batch = torch.tensor([exp.action for exp in exp_batch]).to(self.device)
        values = self.q_net(state_batch)
        return values.masked_select(
            F.one_hot(action_batch, num_classes=self.q_net.num_outputs).bool()
        )

    def reset_target(self):
        self.q_target = deepcopy(self.q_net)
