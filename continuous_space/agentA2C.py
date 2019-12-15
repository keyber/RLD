import torch
from torch import nn
import numpy as np
import torch.optim.lr_scheduler

class BatchA2C_Agent:
    def __init__(self, t_max, action_space, state_space, Q, V, optim_v, optim_q, gamma):
        self.t_max = t_max
        self._t = 0
        self.epoch = 0
        
        self.action_space = action_space
        self.state_space = state_space
        self.Q = Q
        self.V = V
        self.optim_q = optim_q
        self.optim_v = optim_v
        self.scheduler_q = torch.optim.lr_scheduler.ExponentialLR(optim_q, gamma=.999)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(optim_v, gamma=.999)
        
        self.gamma = gamma
        
        self.rewards = []
        self.states = []
        self.actions = []
        
        self.last_state = None
        self.last_action = None
        
        self.baseline_loss = torch.nn.SmoothL1Loss()
        
        self.min_action_count = 1000 #todo dans des états différents et avec xp replay
        self.action_count = np.zeros(action_space, dtype=int)
        self.warmup = True
        
        
    def _update(self, state0, action, state1, reward, done):
        assert not any(x is None for x in (state0, action, state1, reward))
    
        self.states.append(state0)
        self.actions.append(action)
        self.rewards.append(reward)
        
        self._t += 1
        
        if done or self._t == self.t_max:
            losses = self._update_weights(state1, done)
            
            self._t = 0
            self.rewards = []
            self.states = []
            self.actions = []
        
            return losses
        
        return None

    def _update_weights(self, final_state, done):
        """final_state: état dans lequel on est arrivé (et d'où on a pas encore joué)"""
        loss_v = torch.tensor([0.])
        loss_q = torch.tensor([0.])
        
        self.optim_v.zero_grad()
        self.optim_q.zero_grad()
    
        if done:
            # on ne gagnera plus de reward si le jeu est fini
            R = torch.zeros(1)
        else:
            # sinon on estime le reward restant avec V
            R = self.V(final_state)
        
        for i in reversed(range(self._t)):
            R = self.rewards[i] + self.gamma * R
        
            v = self.V(self.states[i])
            q = self.Q(self.states[i])
            
            # detach pour ne pas backpropager pas à travers V
            advantage = R - v.detach()
            
            loss_v += self.baseline_loss(v, R)
            
            loss_q -= torch.log(q[self.actions[i]]) * advantage
        
        loss_q.backward()
        self.optim_q.step()
        self.scheduler_q.step(self.epoch)
        
        loss_v.backward()
        self.optim_v.step()
        self.scheduler_v.step(self.epoch)
        
        self.epoch += 1
        
        return {"loss_q":loss_q.item(), "loss_v":loss_v.item()}
    
    
    def act(self):
        state = self.last_state
        
        action_scores = self.Q(state)
        
        if self.warmup:
            possibilities = np.arange(self.action_space)[self.action_count < self.min_action_count]
            action = np.random.choice(possibilities)
        else:
            action = np.argmax(action_scores.detach().numpy())
            
        self.action_count[action] += 1
        self.warmup = self.warmup and np.any(self.action_count < self.min_action_count)
        
        self.last_action = action
        state_score = self.V(state)
        return action, state_score, action_scores

    def get_result(self, state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        
        losses = self._update(self.last_state, self.last_action, state, reward, done)
        
        if done:
            self.last_state = None
        else:
            self.last_state = state
        
        return losses
    
    def reset(self, state):
        self.last_state = torch.tensor(state, dtype=torch.float32)


class NN_Q(nn.Module):
    def __init__(self, inSize, outSize, layers):
        super().__init__()
        
        l = []
        for x in layers:
            l.append(nn.Linear(inSize, x))
            l.append(nn.LeakyReLU())
            inSize = x
        
        l.append(nn.Linear(inSize, outSize))
        l.append(nn.Softmax(dim=0))
        
        self.layers = nn.Sequential(*l)

    def forward(self, x):
        return self.layers(x)

class NN_V(nn.Module):
    """comme Q sans softmax"""
    def __init__(self, inSize, outSize, layers):
        super().__init__()
        
        l = []
        for x in layers:
            l.append(nn.Linear(inSize, x))
            l.append(nn.LeakyReLU())
            inSize = x
        
        l.append(nn.Linear(inSize, outSize))
        
        self.layers = nn.Sequential(*l)

    def forward(self, x):
        return self.layers(x)
