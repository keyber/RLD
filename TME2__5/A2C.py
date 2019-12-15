import torch
from torch import nn
import numpy as np
import torch.optim.lr_scheduler

class ActorCritic(nn.Module):
    def __init__(self, inSize, outSize, hiddenSize):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(inSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(inSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outSize),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        value = self.critic(state)
        policy_dist = self.actor(state)
        return value, policy_dist

class A2CAgent:

    def __init__(self, t_max, env, n_actions, actor_critic, optimizer, phi, gamma):
        self.t_max = t_max
        self.t = 0

        self.n_actions = n_actions
        self.actor_critic = actor_critic
        self.optimizer = optimizer

        self.phi = phi
        self.gamma = gamma

        self.env = env
        self.rewards = []
        self.values = []
        self.actions = []
        self.log_probs = []
        self.entropy_term = 0

    def _update(self, observation, reward, done):
        s2 = torch.tensor(observation)
        phi2 = self.phi(s2).float()

        self.t += 1

        if done or self.t == self.t_max:
            Qval, _ = self.actor_critic(phi2)
            Qval = Qval.detach().numpy()[0]
            Qvals = np.zeros(len(self.rewards))

            # compute Q values
            for i in range(len(self.rewards) - 1, -1, -1):
                Qval = self.rewards[i] + self.gamma * Qval
                Qvals[i] = Qval

            # update actor critic
            values = torch.tensor(self.values).requires_grad_(True)
            Qvals = torch.tensor(Qvals).requires_grad_(True)
            log_probs = torch.tensor(self.log_probs).requires_grad_(True)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

            # reset buffers
            self.t = 0
            self.log_probs = []
            self.values = []
            self.rewards = []

        return phi2

    def act(self, observation, reward, done):
        phi2 = self._update(observation, reward, done)

        value, policy_dist = self.actor_critic(phi2)
        dist = policy_dist.detach().numpy()
        action = np.random.choice(self.n_actions, p=np.squeeze(dist))

        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_term += entropy


        return action
