from collections import namedtuple
import random
import torch


Experience = namedtuple(
    "Experience", ["state", "action", "reward", "state_next", "done"]
)


class ReplayBuffer:
    """
    Create a circular buffer for experience replay
    :param capacity: maximum buffer size
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.experience = [None] * self.capacity

    def __len__(self):
        return self.size

    def append(self, exp, cast_to_int=False):
        if cast_to_int:
            exp = Experience(
                exp.state.byte(),
                exp.action,
                exp.reward,
                exp.state_next.byte(),
                exp.done,
            )
        self.experience[self.position] = exp
        self.size = min(self.size + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity

    def sample_experience(self, minibatch_size):
        return random.sample(self.experience[: self.size], minibatch_size)
