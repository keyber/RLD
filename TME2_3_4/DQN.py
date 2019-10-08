import torch
from torch import nn
import numpy as np
import copy


class DQN_Agent:
    def __init__(self, env, action_space, Q, optim, loss, C, eps, eps_decay,
                 replay_memory_max_len, replay_memory_n, gamma, phi):
        self.replay_memory = np.empty(replay_memory_max_len, dtype=object)
        self.replay_memory_ind = 0
        self.eps = eps
        self.eps_decay = eps_decay
        self.replay_memory_n = replay_memory_n
        self.gamma = gamma

        self.action_space = action_space
        self.Q = Q
        self.Q_old = copy.deepcopy(self.Q)
        self.phi = phi
        self.optim = optim
        self.loss = loss
        
        self.episode = 0
        self.c = 0
        self.C = C
        self.last_phi_state = None
        self.last_action = None
        
        self.env = env
        
    
    def _update(self, observation, reward, done):
        s2 = torch.tensor(observation)
        phi2 = self.phi(s2)
        
        # ne sauvegarde rien à la première action
        if self.last_phi_state is not None:
            # save
            self.replay_memory[self.replay_memory_ind % len(self.replay_memory)] =(
                self.last_phi_state,
                torch.tensor([self.last_action]),
                torch.tensor([reward]),
                phi2,
                torch.tensor([done], dtype=torch.bool))
            self.replay_memory_ind += 1
            
            # sample for experience replay
            els = np.random.choice(range(min(self.replay_memory_ind, len(self.replay_memory))),
                                   min(self.replay_memory_ind, self.replay_memory_n))
            l = self.replay_memory[els]
            (lphi, la, lr, lphi2, ld) = map(torch.stack, zip(*l))
            la = la.squeeze(1)
            lr = lr.squeeze(1)
            ld = ld.squeeze(1)
            
            # gradient descent
            self.optim.zero_grad()
            phi2 = phi2.detach()
            y_true = torch.where(ld, lr, lr + self.gamma * torch.max(self.Q_old(lphi2), dim=1)[0]) #max retourne (max, argmax)
            y_pred = self.Q(lphi)[range(len(lphi)), la] # advanced indexing (différent de [:, la])
            assert y_true.requires_grad == True
            y_true = y_true.detach()
            y_true.requires_grad = False
            assert y_pred.requires_grad == True
            loss = self.loss(y_pred, y_true)
            loss.backward()
            self.optim.step()

        self.last_phi_state = phi2
        
    
    def act(self, observation, reward, done):
        self._update(observation, reward, done)
        
        self.c += 1
        if self.c % self.C == 0:
            self.Q_old = copy.deepcopy(self.Q)
        
        self.eps *= self.eps_decay
        if np.random.random() < self.eps:
            a = np.random.choice(self.action_space, 1)[0]
        else:
            state_representation = self.last_phi_state
            action_scores = self.Q(state_representation)
            a = np.argmax(action_scores.detach().numpy())
        
        self.last_action = a
        return a


class NN(nn.Module):
    def __init__(self, inSize, outSize, layers):
        super(NN, self).__init__()
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
