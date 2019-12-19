import torch
from torch import nn
import numpy as np
import copy


class DQN_Agent:
    def __init__(self, env, action_space, Q, optim, loss, C, eps, eps_decay,
                 replay_memory_max_len, replay_memory_n, gamma, phi, with_exp_replay=True, with_target_network=True):
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
        self.with_exp_replay = with_exp_replay
        self.with_target_network = with_target_network
    
    def _update(self, observation, reward, done):
        phi2 = self.phi(torch.tensor(observation))
        
        if self.with_exp_replay:
            # save
            self.replay_memory[self.replay_memory_ind % len(self.replay_memory)] = (
                self.last_phi_state,
                torch.tensor([self.last_action]),
                torch.tensor([reward], dtype=torch.float),
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
        else:
            lphi = self.last_phi_state.unsqueeze(0)
            la = torch.tensor([self.last_action]).unsqueeze(0)
            lr = torch.tensor([reward], dtype=torch.float).unsqueeze(0)
            lphi2 = phi2.unsqueeze(0)
            ld = torch.tensor([done], dtype=torch.bool).unsqueeze(0)
        
        # gradient descent
        self.optim.zero_grad()
        phi2 = phi2.detach()
        if self.with_target_network:
            y_true = torch.where(ld, lr, lr + self.gamma *
                                 torch.max(self.Q_old(lphi2), dim=1)[0]) #max retourne (max, argmax)
        else:
            y_true = torch.where(ld, lr, lr + self.gamma *
                                 torch.max(self.Q(lphi2), dim=1)[0])  # max retourne (max, argmax)
        
        y_pred = self.Q(lphi)[range(len(lphi)), la]  # advanced indexing (différent de [:, la])
        assert y_true.requires_grad == True
        y_true = y_true.detach()
        y_true.requires_grad = False
        assert y_pred.requires_grad == True
        loss = self.loss(y_pred, y_true)
        loss.backward()
        self.optim.step()
        
        self.last_phi_state = phi2
    
    def act(self):
        self.eps *= self.eps_decay
        if np.random.random() < self.eps:
            a = np.random.choice(self.action_space, 1)[0]
        else:
            state_representation = self.last_phi_state
            action_scores = self.Q(state_representation)
            a = np.argmax(action_scores.detach().numpy())
        
        self.last_action = a
        return a
    
    def get_result(self, observation, reward, done):
        self._update(observation, reward, done)
        
        self.c += 1
        if self.c % self.C == 0 and self.with_target_network:
            self.Q_old = copy.deepcopy(self.Q)
    
    def reset(self, observation):
        self.last_action = None
        self.last_phi_state = self.phi(torch.tensor(observation))


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
