import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(state_dim, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, action_dim)
        self.min_action = min_action
        self.action_range = max_action - min_action

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 400),
            # nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Sigmoid()
        )


    def forward(self, state):
        state = state.float()
        # a = F.relu(self.l1(state))
        # a = F.relu(self.l2(a))
        #return self.max_action * torch.tanh(self.l3(a))
        raw_a = self.layers(state)
        a = (raw_a * self.action_range) + self.min_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # self.l1 = nn.Linear(state_dim + action_dim, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, 1)


    def forward(self, state, action):
        # q = F.relu(self.l1(torch.cat([state, action], 1)))
        # q = F.relu(self.l2(q))
        # return self.l3(q)
        q = self.layers(torch.cat([state, action], 1))
        return q

class DDPG(object):
    def __init__(self, env, state_dim, action_dim, min_action, max_action, discount=0.99, tau=0.001,
            start_timesteps=1000, expl_noise=.1):
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.actor.float()
        self.actor_target.float()
        self.critic.float()
        self.critic_target.float()

        self.discount = discount
        self.tau = tau

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.last_state = None
        self.last_action = None
        self.start_timesteps = start_timesteps
        self.expl_noise = expl_noise
        self.env = env
        self.min_action = min_action
        self.max_action = max_action
        self.action_dim = action_dim

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_dim, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        self.epsilon = 1.0
        self.epsilon_decay = 1e-6
        
    def act(self, state, warmup=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        if type(state) != torch.Tensor:
            state = torch.tensor(state)
        if warmup:
            action = self.env.action_space.sample()
        else:
            action = self.actor(state).cpu().data.numpy().flatten()
        # action += np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        action += self.noise.sample() * self.epsilon
        action = action.clip(self.min_action, self.max_action)
        
        self.last_action = action
        self.last_state = state.cpu()

        return self.last_action

    def update(self, next_state, reward, done, batch_size):
        # Store data in replay buffer
        if self.last_action is not None:
            self.replay_buffer.add(self.last_state, self.last_action, next_state, reward, done)

        # time to update
        # if self.replay_buffer.size % batch_size == 0 and self.replay_buffer.size > 1:
        if self.replay_buffer.size > batch_size == 0:
                self.train(batch_size)

        self.last_state = next_state

    def train(self, batch_size=100):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.epsilon -= self.epsilon_decay

    def reset_noise(self):
        self.noise.reset()


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state