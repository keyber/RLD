import torch
from torch import nn
import numpy as np

class BatchA2C_Agent:
    def __init__(self, t_max, env, action_space, state_space, Q, V, optim_v, optim_q, phi, gamma):
        self.t_max = t_max
        self.t = 0

        self.action_space = action_space
        self.state_space = state_space
        self.Q = Q
        self.V = V
        self.optim_v = optim_v
        self.optim_q = optim_q

        self.phi = phi
        self.gamma = gamma

        self.env = env
        self.rewards = []
        self.states = []
        self.actions = []

    def _update(self, observation, reward, done):
        s2 = torch.tensor(observation)
        phi2 = self.phi(s2)

        phi2 = phi2.detach()

        self.states.append(phi2)
        self.rewards.append(reward)

        self.t += 1

        if done or self.t == self.t_max:
            self.optim_v.zero_grad()
            self.optim_q.zero_grad()
            if done:
                R = torch.zeros(1)
            else:
                R = self.V(self.states[-1])

            for i in range(self.t-2, -1, -1):
                R = self.rewards[i] + self.gamma * R

                v = self.V(self.states[i])
                q = self.Q(self.states[i])
                # a = q  - v.expand_as(q)
                loss_v = torch.nn.SmoothL1Loss()(v, R)#torch.pow(R - v, 2)
                loss_v.backward(retain_graph=True)

                # loss_a = torch.log(a[self.actions[i]]) * (R - v)
                loss_a = torch.log(q[self.actions[i]]) * (R - v)
                loss_a.backward(retain_graph=True)

            self.optim_v.step()
            self.optim_q.step()
    
    def _update2(self, done):
        if done or self.t == self.t_max:
            self.t = 0
            self.rewards = []
            self.states = []
            self.actions = []


    def act(self, observation, reward, done):
        self._update(observation, reward, done)
        
        state_representation = self.states[-1]
        action_scores = self.Q(state_representation)
        a = np.argmax(action_scores.detach().numpy())

        self._update2(done)
        self.actions.append(a)

        return a


class NN_Q(nn.Module):
    def __init__(self, inSize, outSize, layers):
        super(NN_Q, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.layers.append(nn.Softmax(dim=0))

    def forward(self, x):
        x = self.layers[0](x.float())
        for i in range(1, len(self.layers)):
            x = nn.functional.leaky_relu(x).float()
            x = self.layers[i](x)
        return x

class NN_V(nn.Module):
    def __init__(self, inSize, outSize, layers):
        super(NN_V, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x.float())
        for i in range(1, len(self.layers)):
            x = nn.functional.leaky_relu(x).float()
            x = self.layers[i](x)
        return x
