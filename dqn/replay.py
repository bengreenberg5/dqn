from collections import deque, namedtuple
import numpy as np
import random
import torch


Experience = namedtuple("Experience", ["state", "action", "reward", "state_next", "done"])


# container for experience replay
class ReplayBuffer(deque):
    def __init__(self, maxlen):
        deque.__init__(self, [], maxlen)

    def sample_experience(self, minibatch_size):
        return random.sample(self, minibatch_size)
