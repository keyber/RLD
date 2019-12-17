import torch
import numpy as np
import torch.optim.lr_scheduler
import torch.utils.data

import utils
from utils import assert_numeric
from torch.optim import SGD, Adam

V_PLOT = 1
V_GRAD = 2
V_BENCH = 4
V_WARM = 8
V_EPOCH = 16
V_GRAD_NORM = 32

class BatchA2C_Agent:
    def __init__(self, t_max, action_space, state_space, Q, V, gamma=.99, writer=None, verbose=0):
        self.t_max = t_max
        self._t = 0
        self.epoch = 0
        
        self.action_space = action_space
        self.state_space = state_space
        self.Q = Q
        self.V = V
        self.optim_q = Adam(Q.parameters(), lr=1e-2, weight_decay=1e-1)
        self.optim_v = Adam(V.parameters(), lr=1e-2)
        self.scheduler_q = torch.optim.lr_scheduler.ExponentialLR(self.optim_q, gamma=.999)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optim_v, gamma=.999)
        
        self.current_trajectories = [[]]
        self.old_traj = {}
        self.mean_traj_r = []
        self.gamma = gamma
        
        self.last_state = None
        self.last_action = None
        
        self.baseline_loss = torch.nn.SmoothL1Loss()
        
        self.v_warmup_loss_convergence_eps = 1e-1
        self.min_action_count = 100 #todo dans des états différents et avec xp replay
        self.action_count = np.zeros(action_space, dtype=int)
        self.warmup = True
        
        self.q_backprop_has_started = False
        
        self.benchmarked_trajectories = None
        self.space_bounds = None
        
        self.writer = writer
        self.verbose = verbose
        self.animator = utils.Anim(action_space, nb_frames=200)
    
    def _get_dl_from_memory(self, mem, done=True, final_state=None, batch_size=9999, get_mean_traj_reward=False):
        """done dit si la dernière trajectoire est finie
        si ce n'est pas le cas, la trajectoire est actuellement dans l'état final state"""
        assert done or (final_state is not None)
        list_i = []
        list_s = []
        list_a = []
        list_r = []
        
        traj_reward = []
        for i_mem, trajectory in enumerate(mem):
            r_real = 0
            
            if i_mem < len(self.current_trajectories) - 1 or done:
                # on ne gagnera plus de reward si le jeu est fini
                r_disc = torch.tensor(0.)
            else:
                # sinon on estime le reward restant avec V
                r_disc = self.V(final_state)
        
            for i_traj, (state, action, reward, _state2) in reversed(list(enumerate(trajectory))):
                r_real += reward
                r_disc = reward + self.gamma * r_disc
                list_i.append(i_traj)
                list_s.append(state)
                list_a.append(action)
                list_r.append(r_disc)
            
            traj_reward.append(r_real)
        
        dl = torch.utils.data.DataLoader(list(zip(list_i, list_s, list_a, list_r)), batch_size=batch_size)
        
        if get_mean_traj_reward:
            return dl, sum(traj_reward)/len(traj_reward)
        
        return dl
    
    def _end_warmup_learn_v(self):
        print("warmup", len(self.current_trajectories), "trajectories")
        
        optim = Adam(self.V.parameters(), lr=1e-1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.99)
        
        dl = self._get_dl_from_memory(self.current_trajectories, done=True)
        
        epoch = 0
        gbl_step = 0
        loss_diff = self.v_warmup_loss_convergence_eps 
        loss_old = [loss_diff * i for i in range(5)]
        loss_mean = 0
        while loss_diff >= self.v_warmup_loss_convergence_eps:
            loss_mean = []
            for i, s, _a, r in dl:
                assert not s.requires_grad and not r.requires_grad
                pred = self.V(s)
                loss = self.baseline_loss(pred, r)
                loss_mean.append(loss.item())
                assert_numeric(loss)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if self.writer is not None:
                    self.writer.add_scalar("loss_v_warmup", loss, gbl_step)
                    if self.verbose & V_GRAD_NORM:
                        grad = {name:torch.mean(x.grad) for (name, x) in self.V.named_parameters() if hasattr(x, "grad")}
                        self.writer.add_scalars("grad_v_warmup", grad, gbl_step)
                    gbl_step += 1
            
            loss_mean = np.mean(loss_mean)
            loss_old.pop(0)
            loss_old.append(loss_mean)
            loss_diff = sum(abs(loss_old[i] - loss_old[i+1]) for i in range(len(loss_old)-1)) / len(loss_old)
            epoch += 1
            scheduler.step(epoch)
        
        assert_numeric(self.V)
        
        print("warmup V ", epoch, " epochs, MAE=", loss_mean,
              " (eps=", self.v_warmup_loss_convergence_eps, ")\n", sep="")
    
    def _update(self, state0, action, state1, reward, done):
        assert not any(x is None for x in (state0, action, state1, reward))
        
        self.current_trajectories[-1].append((state0, action, reward, state1))
                
        # récolte des transitions aléatoirement
        if self.warmup:
            self.warmup = np.any(self.action_count < self.min_action_count)
        
        
        # quand le warmup vient de finir et qu'on est à la fin d'une partie
        if not self.q_backprop_has_started and not self.warmup and done :
            # entraîne v
            self._end_warmup_learn_v()
            
            # démarre l'entraînement de q
            self.q_backprop_has_started = True
        
        
        if self.q_backprop_has_started and (done or self._t == self.t_max):
            # crée le dataloader
            res = self._get_dl_from_memory(self.current_trajectories, get_mean_traj_reward=True,
                                           done=done, final_state=state1)
            self.old_traj[self.epoch], r = res
            self.mean_traj_r.append((r, self.epoch))
            
            # plot V et Q
            if self.verbose & V_PLOT:
                self.animator.log(self.old_traj, self.V, self.Q)
                
            # entraîne V et Q
            self._update_weights(done, state1)
            
            # supprime les mauvaises trajectoires
            n_kept = 10 + self.epoch // 10
            self.mean_traj_r = sorted(self.mean_traj_r, reverse=True)
            for reward_, epoch_ in self.mean_traj_r[n_kept:]:
                del self.old_traj[epoch_]
            self.mean_traj_r = self.mean_traj_r[:n_kept]
            
            # remise à zéro des trajectoire courantes
            self._t = 0
            self.current_trajectories = [[]]
            self.epoch += 1
        else:
            self._t += 1
            if done:
                # commence une nouvelle trajectoire
                self.current_trajectories.append([])
    
    def _update_q(self, dl):
        l_loss_q = []
        
        for i, state, action, reward, in dl:
            assert not state.requires_grad
            v = self.V(state).detach()
            q = self.Q(state)
            advantage = reward - v
            loss_q = -torch.log(q[range(len(q)), action]) * advantage
        
            assert loss_q.shape == (i.shape[0],)
        
            if self.verbose & V_BENCH and self.epoch != 0 and self.benchmarked_trajectories is not None:
                print("ind \t\t state \t\t\t rew \t\t\t v(s) \t\t advant(s) \t\t\t q \t\t\t\t\t action \t\t loss")
                for ii, ss, rr, vv, ad, qq, ac, ll in zip(i, state, reward, v, advantage, q.data, action, loss_q.data):
                    print(ii, ss[2] - ss[0], rr, vv, ad, qq, ac, ll)
        
            loss_q = loss_q.mean()
            l_loss_q.append(loss_q.item())
            assert_numeric(loss_q)
            self.optim_q.zero_grad()
            loss_q.backward()
            self.optim_q.step()
        
        return sum(l_loss_q)/len(l_loss_q)
    
    def _update_v(self, traj_dl):
        l_loss_v = []
        self.optim_v.zero_grad()
        
        for dl in traj_dl.values():
            for _i, state, action, reward, in dl:
                assert not state.requires_grad
                loss_v = self.baseline_loss(self.V(state), reward)
                l_loss_v.append(loss_v.item())
            
                assert_numeric(loss_v)
                self.optim_v.zero_grad()
                loss_v.backward()
                self.optim_v.step()
        
        return sum(l_loss_v)/len(l_loss_v)
    
    def _update_weights(self, done, final_state):
        if self.verbose & V_EPOCH:
            print("update epoch", self.epoch)
        
        dl = self._get_dl_from_memory(self.current_trajectories, done, final_state)
        
        loss_q = self._update_q(dl) #todo Importance Sampling ou seulement la dernière trajectoire
        
        if self.verbose & V_BENCH and self.benchmarked_trajectories is not None:
            print("trajectories")
            for traj in self.benchmarked_trajectories:
                utils.show_trajectory(self.Q, self.V, traj)
        
        loss_v = self._update_v(self.old_traj)
        
        self.scheduler_q.step(self.epoch)
        self.scheduler_v.step(self.epoch)
        
        assert_numeric(self.Q, self.V)
                
        if self.writer is not None:
            losses = {"loss_q": loss_q, "loss_v": loss_v}
            self.writer.add_scalars("loss", losses, self.epoch)
    
    def act(self):
        state = self.last_state
        
        if self.warmup:
            possibilities = np.arange(self.action_space)[self.action_count < self.min_action_count]
            action = np.random.choice(possibilities)
        else:
            action_scores = self.Q(state.unsqueeze(0))
            action = np.argmax(action_scores.detach().numpy())
            
        self.action_count[action] += 1
        
        self.last_action = action
        return action
    
    def get_result(self, state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        
        self._update(self.last_state, self.last_action, state, reward, done)
        
        if done:
            self.last_state = None
        else:
            self.last_state = state
    
    def reset(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.last_state = state
    
