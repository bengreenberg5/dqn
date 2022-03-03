from copy import deepcopy
import os

import torch
import torch.nn as nn


class QNet(nn.Module):

    def __init__(self, num_outputs, layers, device):
        super().__init__()
        self.num_outputs = num_outputs
        self.layers = layers.to(device)
        self.device = device

    def forward(self, x):
        if x.device is not self.device:
            x = x.to(self.device)
        return self.layers(x)

    def get_best_action(self, state):
        return self.forward(state).argmax(dim=-1)

    def est_values(self, states, actions, grad=True):
        if grad:
            return self.forward(states).gather(1, actions.unsqueeze(-1)).squeeze()
        else:
            return self.forward(states).detach().gather(1, actions.unsqueeze(-1)).squeeze()

    def save(self, dirname, fname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save(self.state_dict(), f"{dirname}/{fname}.pt")

    def load(self, dirname, fname, checkpoint):
        assert os.path.exists(dirname), f"directory {dirname} does not exist"
        checkpoint = str(checkpoint).zfill(7)
        self.load_state_dict(torch.load(f"{dirname}/{checkpoint}/{fname}"))


class LinearQNet(QNet):

    def __init__(self, num_inputs, num_outputs, linear_layers=None, device="cpu"):
        super().__init__(num_outputs, linear_layers, device)
        self.num_inputs = num_inputs
        if not linear_layers:
            layers = [50, 50]
        layer_list = [nn.Linear(num_inputs, linear_layers[0]), nn.ReLU()]
        for i in range(1, len(linear_layers)):
            layer_list.append(nn.Linear(linear_layers[i - 1], linear_layers[i]))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(linear_layers[-1], self.num_outputs))
        self.layers = nn.Sequential(*layer_list).to(device)

    def forward(self, x):
        super().forward(x)


class ConvQNet(QNet):

    def __init__(self, num_outputs, device="cpu"):
        layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, num_outputs)
        ).to(device)
        super().__init__(num_outputs, layers, device)

    def forward(self, x):
        super().forward(x)


class DQNAgent:

    @gin.configurable
    def __init__(
        self,
        network_type="linear",
        learning_rate=1e-4,
        momentum=0.95,
        discount_factor=0.99,
        linear_layers=None,
        device="cpu",
    ):
        """
        :param network_type: "linear" or "conv"
        :param learning_rate:
        :param momentum:
        :param discount_factor:
        :param linear_layers: Sizes of linear layers; ignored for conv net
        :param device:
        """
        assert network_type in ("linear", "conv"), f"unknown network type `{network_type}`"

        self.network_type = network_type
        self.discount_factor = discount_factor
        self.device = device

        if network_type == "linear":
            self.q_act = LinearQNet(num_inputs=4, num_outputs=4, linear_layers=linear_layers, device=device)
        elif network_type == "conv":
            self.q_act = ConvQNet(num_outputs=4, device=device)
        self.q_eval = deepcopy(self.q_act)
        self.optimizer = torch.optim.RMSprop(
            self.q_act.parameters(), lr=learning_rate, momentum=momentum
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def apply_grad(self):
        self.optimizer.step()

    def get_best_action(self, state, target=False):
        if not target:
            return self.q_act.get_best_action(state)
        else:
            return self.q_eval.get_best_action(state)

    def est_values(self, states, actions, target=False):
        if not target:
            return self.q_act.est_values(states, actions)
        else:
            return self.q_eval.est_values(states, actions)

    def update_target(self):
        self.q_eval = deepcopy(self.q_act)

    def save(self, dirname, checkpoint):
        self.q_act.save(f"{dirname}/{checkpoint}/", "q_act.pt")
        self.q_eval.save(f"{dirname}/{checkpoint}/", "q_eval.pt")
