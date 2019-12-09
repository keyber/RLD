import torch
from torch import nn
import numpy as np
import torch.optim.lr_scheduler

class BatchA2C_Agent:
    def __init__(self, t_max, env, action_space, state_space, Q, V, optim_v, optim_q, gamma):
        self.t_max = t_max
        self._t = 0
        self.epoch = 0
        
        self.action_space = action_space
        self.state_space = state_space
        self.Q = Q
        self.V = V
        self.optim_v = optim_v
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(optim_v, gamma=.999)
        self.optim_q = optim_q
        
        self.gamma = gamma
        
        self.env = env
        self.rewards = []
        self.states = []
        self.actions = []
        
        self.last_state = None
        self.last_action = None
        
        self.baseline_loss = torch.nn.SmoothL1Loss()
        
    def _update(self, reward, done):    
        assert (self.last_state is None) == (reward is None)
    
        if reward is not None:
            self.states.append(self.last_state)
            self.actions.append(self.last_action)
            self.rewards.append(reward)
            
            self._t += 1
        
        if done or self._t == self.t_max:
            losses = self._update_weights(done)
            
            self._t = 0
            self.rewards = []
            self.states = []
            self.actions = []
        
            return losses
        
        return None

    def _update_weights(self, done):
        loss_v = 0
        loss_q = 0
        
        self.optim_v.zero_grad()
        self.optim_q.zero_grad()
    
        if done:
            R = torch.zeros(1)
        else:
            R = self.V(self.states[-1])
    
        for i in range(self._t - 1, -1, -1):
            R = self.rewards[i] + self.gamma * R
        
            v = self.V(self.states[i])
            q = self.Q(self.states[i])
            advantage = R - v
            # print(i, "%.2f"%R.item(), "%.2f"%v, "%.2f"%advantage.item())
        
            loss_v += self.baseline_loss(v, R)
        
            loss_q -= torch.log(q[self.actions[i]]) * advantage
        
        # print()
        loss_tot = loss_q + loss_v
        loss_tot.backward()
    
        self.optim_v.step()
        self.optim_q.step()
        self.scheduler_v.step(self.epoch)
        self.epoch += 1
        
        return {"loss_q":loss_q.item(), "loss_v":loss_v.item()}
    
    
    def act(self, state, reward, done):
        losses = self._update(reward, done)
        
        state = torch.tensor(state)
        
        action_scores = self.Q(state)
        action = np.argmax(action_scores.detach().numpy())
        
        self.last_state = state
        self.last_action = action
        return action, losses


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
