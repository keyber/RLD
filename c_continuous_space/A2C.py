import torch
from torch import nn
import numpy as np
import torch.optim.lr_scheduler
import torch.utils.data

class BatchA2C_Agent:
    def __init__(self, t_max, action_space, state_space, Q, V, optim_v, optim_q, gamma, writer=None):
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
        
        self.memory = [[]]
        self.gamma = gamma
        
        self.last_state = None
        self.last_action = None
        
        self.baseline_loss = torch.nn.SmoothL1Loss()
        
        self.loss_min = 1.0
        self.v_warmup_epoch = 10
        self.min_action_count = 100 #todo dans des états différents et avec xp replay
        self.action_count = np.zeros(action_space, dtype=int)
        self.warmup = True
        
        self.q_backprop_has_started = False

        self.writer = writer
        
    def _end_warmup_learn_v(self):
        """final_state: état dans lequel on est arrivé (et d'où on n'a pas encore joué)"""
        
        print("end warmup", len(self.memory), "trajectories")
        
        self.optim_v.zero_grad()
    
        list_s = []
        list_r = []
        
        with torch.no_grad():
            for trajectory in self.memory:
                R = torch.zeros(1)
                for (state, _action, reward, _state2) in reversed(trajectory):
                    R = reward + self.gamma * R
                    list_r.append(R)
                    list_s.append(state)
        
        list_s = torch.stack(list_s)
        list_r = torch.stack(list_r)
        
        dl = torch.utils.data.DataLoader(zip(list_s, list_r), batch_size=32)
        
        loss = 1
        epoch = 0
        gbl_step = 0
        while loss > self.loss_min:
            for s, v in dl:
                pred = self.V(s)
                loss = self.baseline_loss(pred, v)
                loss.backward()
                self.optim_v.step()
                
                if self.writer is not None:
                    self.writer.add_scalar("loss_v_warmup", loss, gbl_step)
                    gbl_step += 1
            
            epoch += 1
            self.scheduler_v.step(epoch)
    
    def _update(self, state0, action, state1, reward, done):
        assert not any(x is None for x in (state0, action, state1, reward))
        
        self.memory[-1].append((state0, action, reward, state1))
                
        # récolte des transitions aléatoirement
        if self.warmup:
            self.warmup = np.any(self.action_count < self.min_action_count)
        
        
        # quand le warmup vient de finir et qu'on est à la fin d'une partie
        if not self.q_backprop_has_started and not self.warmup and done :
            # entraîne v
            self._end_warmup_learn_v()
            
            # démarre l'entraînement de q
            self.q_backprop_has_started = True
        
        
        # entraîne v et q
        if self.q_backprop_has_started and (done or self._t == self.t_max):
            losses = self._update_weights(state1, done)
            
            self._t = 0
            
            # vide toute la mémoire car q a changé todo Importance Sampling
            self.memory = [[]]
            
            return losses
        
                
        self._t += 1
        
        if done:
            # commence une nouvelle trajectoire
            self.memory.append([])
        
        return None
    
    def _update_weights(self, final_state, done):
        """final_state: état dans lequel on est arrivé (et d'où on n'a pas encore joué)
        (toutes les anciennes trajectoires sont finies)"""

        for i_traj, trajectory in enumerate(self.memory):
            loss_v = torch.tensor([0.])
            loss_q = torch.tensor([0.])
            
            self.optim_v.zero_grad()
            self.optim_q.zero_grad()
            
            advantage_list = []
            
            if i_traj < len(self.memory) - 1 or done:
                # on ne gagnera plus de reward si le jeu est fini
                R = torch.zeros(1)
            else:
                # sinon on estime le reward restant avec V
                R = self.V(final_state)
            
            for state, action, reward, _state2 in reversed(trajectory):
                R = reward + self.gamma * R
                
                v = self.V(state)
                q = self.Q(state)
                
                # detach pour ne pas backpropager à travers V
                advantage = R - v.detach()
                
                loss_v += self.baseline_loss(v, R)
                
                loss_q -= torch.log(q[action]) * advantage
                
                advantage_list.append(advantage)
    
                # print("update, traj", self.epoch)
                # print("action %.2f"%self.actions[i], "  R %.2f"% R.item(),
                #       "  v %.2f"%v.item(), "  a %.2f"%advantage.item(), "  q", q.detach().numpy())
                # print("loss", loss_q)
                # loss_q.backward()
                # self.optim_q.step()
                # loss_q = torch.tensor([0.])
                # for s in trajectoire_exemple:
                #     state_v, actions = self.get_values(s)
                #     print(state_v.item(), actions.detach().numpy())
            # print()
                
            
            loss_q.backward()
            self.optim_q.step()
            self.scheduler_q.step(self.epoch)
            
            loss_v.backward()
            self.optim_v.step()
            self.scheduler_v.step(self.epoch)
        
        self.epoch += 1
        
        assert not any(torch.any(torch.isnan(p.data)) for p in (list(self.Q.parameters()) + list(self.V.parameters())))
        
        if self.writer is not None:
            # noinspection PyUnboundLocalVariable
            losses = {"loss_q": loss_q.item(), "loss_v": loss_v.item(),
                      "advantage": sum(advantage_list) / len(advantage_list)}
            self.writer.add_scalars("loss", losses, self.epoch)
    
    
    def act(self):
        state = self.last_state
        
        action_scores = self.Q(state)
        
        if self.warmup:
            possibilities = np.arange(self.action_space)[self.action_count < self.min_action_count]
            action = np.random.choice(possibilities)
        else:
            action = np.argmax(action_scores.detach().numpy())
            
        self.action_count[action] += 1
        
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
    
    def get_values(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float32)

        return self.V(state), self.Q(state)

class NN_Q(nn.Module):
    def __init__(self, inSize, outSize, layers, eps=1e-3):
        super().__init__()
        self.eps = eps

        l = []
        for x in layers:
            l.append(nn.Linear(inSize, x))
            l.append(nn.LeakyReLU())
            inSize = x
        
        l.append(nn.Linear(inSize, outSize))
        l.append(nn.Softmax(dim=0))
        
        self.layers = nn.Sequential(*l)

    def forward(self, x):
        res = self.layers(x)
        
        zeroes = res==0
        if torch.any(zeroes):
            n_zero = torch.sum(zeroes).item()
            res = res * (1 - n_zero * self.eps)
            res[zeroes] += self.eps
            assert torch.sum(res) == 1.0
        
        return res

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
