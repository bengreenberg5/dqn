from collections import deque, namedtuple
import numpy as np
import random
import torch


Experience = namedtuple(
    "Experience", ["state", "action", "reward", "state_next", "done"]
)


# container for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.experience = [None] * self.capacity

    def __len__(self):
        return self.size

    def append(self, exp):
        self.experience[self.position] = exp
        self.size = min(self.size + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity

    def sample_experience(self, minibatch_size):
        return random.sample(self.experience[: self.size], minibatch_size)
