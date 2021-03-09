import random
import numpy as np
import torch
from collections import namedtuple

from .sum_tree import SumTree

class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device='cpu'):
        self.action_size = action_size
        self.memory = SumTree(buffer_size) # <- deque(maxlen=buffer_size)
        self.memory_buffer = []  # store experiences without error
        self.batch_size = batch_size

        experience_field_names = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=experience_field_names)
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory_buffer.append(e) # <- self.memory.append(e)

    def sample(self):
        experiences = [] # random.sample(self.memory, k=self.batch_size)
        indices = []
        probs = []

        gap = self.batch_size - len(self.memory_buffer)
        if gap:
            for _ in range(gap):
                s = random.uniform(0, self.memory.total())
                idx, p, e = self.memory.get(s)
                experiences.append(e)
                indices.append(idx)
                probs.append(p / self.memory.total())

        for e in self.memory_buffer:
            experiences.append(e)
            idx = self.memory.add(0.0, e)
            indices.append(idx)
            probs.append(1 / len(self))

        self.memory_buffer.clear()

        np_s = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(np_s).float().to(self.device)

        np_a = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(np_a).float().to(self.device)

        np_r = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(np_r).float().to(self.device)

        np_n = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(np_n).float().to(self.device)

        np_d = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        dones = torch.from_numpy(np_d).float().to(self.device)

        return (states, actions, rewards, next_states, dones, indices, probs)


    def update(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.memory.update(idx, p)


    def __len__(self):
        return max(len(self.memory), len(self.memory_buffer)) # <- len(self.memory)
