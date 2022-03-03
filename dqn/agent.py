from copy import deepcopy
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, layers=None, device="cpu"):
        """
        Create a Q-network
        :param num_inputs:
        :param num_outputs:
        :param layers: Sizes of linear layers. If None, set up default Atari network:
        :param device: Train with "cpu" or "cuda"
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if not layers:
            pass  # TODO make atari
        else:
            layer_list = [nn.Linear(num_inputs, layers[0])]
            for i in range(1, len(layers)):
                layer_list.append(nn.Linear(layers[i - 1], layers[i]))
                layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(layers[-1], self.num_outputs))
            self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self, dirname, target=False):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        fname = "q_net.pt" if not target else "q_target.pt"
        torch.save(self.q_net.state_dict(), f"{dirname}/{fname}.pt")

    def load(self, dirname, checkpoint, target=False):
        assert os.path.exists(dirname), f"directory {dirname} does not exist"
        checkpoint = str(checkpoint).zfill(7)
        fname = "q_net.pt" if not target else "q_target.pt"
        self.q_net.load_state_dict(torch.load(f"{dirname}/{checkpoint}/{fname}"))


class DQNAgent:
    def __init__(
        self,
        env_name,
        learning_rate=1e-4,
        momentum=0.95,
        discount_factor=0.99,
        device="cpu",
    ):
        """
        TODO
        :param env_name:
        :param learning_rate:
        :param momentum:
        :param discount_factor:
        :param device:
        """
        self.q_net = None  # TODO
        self.q_target = None  # TODO
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(), lr=learning_rate, momentum=momentum
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net = QNet(env_name).to(self.device)
        self.q_target = deepcopy(self.q_net)

        torch.save(self.q_target.state_dict(), f"{dirname}/q_target.pt")

    def get_action(self, state):
        rewards = self.q_net(state.float().to(self.device))
        return torch.argmax(rewards).item()

    def zero_grad(self):
        pass

    def q_target_estimate(self, state_nexts):
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

    def q_value_estimate(self, states):  # TODO rename
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
