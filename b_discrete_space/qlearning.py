import itertools
import random
import numpy as np


class QLearningAgent:
    def __init__(self, action_space, default=0, lr=.5, gamma=.99, eps=1., eps_decay=1.):
        self.default = default
        self.q = {}  # état, action : valeur
        
        self.last_action = None
        self.last_state = None
        self.lr = lr
        self.gamma = gamma
        self.action_space = action_space
        self.eps = eps
        self.eps_decay = eps_decay
    
    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)
        
        if done:
            m = 0
        else:
            m = -np.inf
            for action2 in self.action_space:
                m = max(m, self.q.get((state2, action2), self.default))
        
        val += self.lr * (reward + self.gamma * m - val)
        
        self.q[(state, action)] = val
    
    def act(self):
        action = self.epsilon_greedy(self.last_state, self.eps)
        self.last_action = action
        return action
    
    def get_result(self, state, reward, done):
        self._updateQ(self.last_state, self.last_action, state, reward, done)
        self.eps *= self.eps_decay
        self.last_state = state
    
    def reset(self, state):
        self.last_state = state
    
    def epsilon_greedy(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.action_space, 1)[0]
        
        best_val = -np.inf
        best_action = None
        
        for action in self.action_space:
            val = self.q.get((state, action), self.default)
            if val > best_val:
                best_val = val
                best_action = action
        
        return best_action
    
    def get_policy(self, states):
        return {state: max(self.action_space, key=lambda a:self.q.get((state, a), self.default))
                for state in states}


class SarsaAgent(QLearningAgent):
    def __init__(self, action_space, default=0, lr=.5, gamma=.99, eps=1.):
        super().__init__(action_space, default, lr, gamma, eps)
        self.next_action = None
    
    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)
        
        if done:
            val += self.lr * reward
        else:
            self.next_action = self.epsilon_greedy(state2, self.eps)
            val += self.lr * (reward + self.gamma * self.q.get((state2, self.next_action), self.default) - val)
        
        self.q[(state, action)] = val
    
    def act(self):
        if self.next_action is None:
            action = self.epsilon_greedy(self.last_state, self.eps)
        else:
            action = self.next_action
        
        self.last_action = action
        
        return action
    
    def reset(self, state):
        self.last_state = state
        self.next_action = None


class DynaQAgent(QLearningAgent):
    def __init__(self, action_space, state_space, lr_R, lr_P, k, default=0, lr=.5, gamma=.99, eps=1.):
        super().__init__(action_space, default, lr, gamma, eps)
        self.state_space = state_space
        self.lr_R = lr_R
        self.lr_P = lr_P
        self.R_hat = {}
        self.P_hat = {}
        self.k = k
    
    def _updateMDP(self, state, action, state2, reward):
        val_R = self.R_hat.get((state, action, state2), self.default)
        val_P = self.R_hat.get((state, action, state2), self.default)
        self.R_hat[(state, action, state2)] = val_R + self.lr_R * (reward - val_R)
        self.P_hat[(state, action, state2)] = val_P + self.lr_P * (1 - val_P)
        
        for state3 in self.state_space:
            if state3 != state2:
                val_P = self.P_hat.get((state, action, state3), self.default)
                self.P_hat[(state, action, state3)] = val_P + self.lr_P * (-val_P)
        
        state_action_pairs = list(itertools.product(self.state_space, self.action_space))
        sampled_pairs = random.sample(state_action_pairs, self.k)
        
        for (s, a) in sampled_pairs:
            val_q = self.q.get((s, a), self.default)
            sum_ = 0
            
            for s2 in self.state_space:
                val_p = self.P_hat.get((s, a, s2), self.default)
                val_r = self.R_hat.get((s, a, s2), self.default)
                m = -np.inf
                for a2 in self.action_space:
                    m = max(m, self.q.get((s2, a2), self.default))
                
                # todo à vérifier (je ne sais pas si les parenthèses sont bien placées dans le slide)
                # avec val_p * (val_r + self.gamma * m) - val_q (comme sur le slide), les valeurs explosent
                sum_ += val_p * (val_r + self.gamma * m - val_q)
            
            self.q[(s, a)] = val_q + self.lr * sum_
    
    def get_result(self, state, reward, done):
        self._updateMDP(self.last_state, self.last_action, state, reward)
        super().get_result(state, reward, done)
