import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        fields = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=fields)
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        np_s = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(np_s).float().to(self.device)

        np_a = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(np_a).float().to(self.device)

        np_r = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(np_r).float().to(self.device)

        np_n = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(np_n).float().to(self.device)

        rp_dones = [e.done for e in experiences if e is not None]
        np_d = np.vstack(rp_dones).astype(np.uint8)
        dones = torch.from_numpy(np_d).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
