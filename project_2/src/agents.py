import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from .replay_buffer import ReplayBuffer
from .models import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        # self.size = size  # added for sample
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        w = self.sigma * np.array([random.random() for i in range(len(x))])
        # w = self.sigma * np.random.binomial(num_agents, self.size)
        dx = self.theta * (self.mu - x) + w
        self.state = x + dx
        return self.state


class DDPGAgent():
    """Deep Deterministic Policy Gradient Agent"""

    def __init__(self, state_size, action_size, random_seed,
                 buffer_size, batch_size, gamma, tau,
                 lr_actor, lr_critic, weight_decay,
                 update_every, update_times,
                 eps_start=1.0, eps_end=0.01, eps_decay=1e-6):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.update_times = update_times

        # Epsilon Decay
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # initialize Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=lr_actor
        )

        # initialize Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=lr_critic,
            weight_decay=weight_decay
        )

        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)

        self.step_count = 0


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.step_count += 1
        self.step_count %= self.update_every

        if len(self.memory) > self.batch_size and self.step_count == 0:
            for _ in range(self.update_times):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.eps * self.noise.sample()

        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma):
        # --- update critic ---
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --- update actor ---
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target newtorks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # update noise
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def soft_update(self, local_model, target_model, tau):
        target_params = target_model.parameters()
        local_params = local_model.parameters()
        for target, local in zip(target_params, local_params):
            target.data.copy_(tau * local.data + (1.0 - tau) * target.data)


    def save(self, actor_local_path, actor_target_path,
             critic_local_path, critic_target_path):
        torch.save(self.actor_local.state_dict(), actor_local_path)
        torch.save(self.actor_target.state_dict(), actor_target_path)
        torch.save(self.critic_local.state_dict(), critic_local_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)


    def load(self, actor_local_path, actor_target_path,
             critic_local_path, critic_target_path):
        self.actor_local.load_state_dict(torch.load(actor_local_path))
        self.actor_target.load_state_dict(torch.load(actor_target_path))

        self.critic_local.load_state_dict(torch.load(critic_local_path))
        self.critic_target.load_state_dict(torch.load(critic_target_path))
