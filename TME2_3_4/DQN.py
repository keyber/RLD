import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import copy

class DQN_Agent:
    def __init__(self, env, action_space, sizeIn, sizeOut, T, C, eps, replay_memory_max_len, batch_size, gamma, phi):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.replay_memory = np.empty(replay_memory_max_len, dtype=object)
        self.replay_memory_ind = 0
        self.T = T
        self.eps = eps
        self.batch_size = batch_size
        self.gamma = gamma

        self.action_space = action_space
        self.Q = NN(self.sizeIn, self.sizeOut)
        self.Q_old = copy.deepcopy(self.Q)
        self.phi = phi
        self.optim = Adam(self.Q.parameters())
        
        self.episode = 0
        self.t = 0
        self.c = 0
        self.C = C
        self.last_phi_state = None
        self.last_action = None
        
        self.env = env
        
        
    
    def _update(self, observation, reward, done):
        s2 = self.env.state2str(observation)
        phi2 = self.phi(s2)
        
        # save
        self.replay_memory[self.replay_memory_ind % len(self.replay_memory)] = None
        self.replay_memory[self.replay_memory_ind % len(self.replay_memory)] = (self.last_phi_state, self.last_action, reward, phi2, done)
        self.replay_memory_ind += 1
        
        # sample for experience replay
        els = np.random.choice(range(min(self.replay_memory_ind, len(self.replay_memory)), self.batch_size))
        l = self.replay_memory[els]
        print(l)
        (phi, a, r, phi2, d) = map(torch.Tensor, zip(*l))
        
        # gradient descent
        self.optim.zero_grad()
        phi2 = phi2.detach()
        y_true = torch.where(d, r, r + self.gamma * torch.max(self.Q_old(phi2), dim=1))
        y_pred =  self.Q(phi)
        assert y_true.requires_grad == False
        assert y_pred.requires_grad == True
        loss = nn.SmoothL1Loss(y_true - y_pred)
        loss.backward()
        self.optim.step()

        self.last_phi_state = phi2
        
    
    def act(self, observation, reward, done):
        self._update(observation, reward, done)
        
        # self.t += 1 # fixme t inutilisé ?
        # if self.t == self.T:
        #     self.t = 0
        
        self.c += 1
        if self.c % self.C == 0:
            self.Q_old = copy.deepcopy(self.Q)
            
        if np.random.random() < self.eps:
            a = np.random.choice(self.action_space, 1)[0]
        else:
            state_representation = self.last_phi_state
            action_scores = self.Q(state_representation)
            a = np.argmax(action_scores)
        
        self.last_action = a
        return a
            
            
                

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x
    
    #todo target.detach()
