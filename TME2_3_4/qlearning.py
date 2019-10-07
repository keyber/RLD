import itertools
import random

import numpy as np

class QLearningAgent:
    def __init__(self, env, action_space, default, alpha, gamma, eps):
        # état, action : valeur
        self.default = default
        self.q = {}  # todo
        
        self.env = env
        self.last_action = None
        self.last_state = None
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.eps = eps
    
    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)
        
        if done:  # todo à vérifier
            m = 0
        else:
            m = -np.inf
            for action2 in self.action_space:
                m = max(m, self.q.get((state2, action2), self.default))
        
        val += self.alpha * (reward + self.gamma * m - val)
        
        self.q[(state, action)] = val
    
    def act(self, observation, reward, done):
        state2 = self.env.state2str(observation)
        self._updateQ(self.last_state, self.last_action, state2, reward, done)
        
        action2 = self.epsilon_greedy(state2, self.eps)
        
        self.last_state = state2
        self.last_action = action2
        
        return self.last_action
    
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


class SarsaAgent(QLearningAgent):
    def __init__(self, env, action_space, default, alpha, gamma, eps, cst):
        super().__init__(env, action_space, default, alpha, gamma, eps)
        self.cst = cst
    
    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)
        
        if done:  # todo à vérifier
            val += self.alpha * reward
        else:
            action2 = self.epsilon_greedy(state, self.eps)
            val += self.alpha * (reward + self.gamma * self.q.get((state2, action2), self.default) - val)
        
        self.q[(state, action)] = val
    

class DynaQAgent(QLearningAgent):
    def __init__(self, env, action_space, state_space, default, alpha, gamma, eps,
                 alpha_R, alpha_P, k):
        super().__init__(env, action_space, default, alpha, gamma, eps)
        self.state_space = state_space
        self.alpha_R = alpha_R
        self.alpha_P = alpha_P
        self.R_hat = {}
        self.P_hat = {}
        self.k = k

    def _updateMDP(self, state, action, state2, reward):
        val_R = self.R_hat.get((state, action, state2), self.default)
        val_P = self.R_hat.get((state, action, state2), self.default)
        self.R_hat[(state, action, state2)] = val_R + self.alpha_R * (reward - val_R)
        self.P_hat[(state, action, state2)] = val_P + self.alpha_P * (1 - val_P)

        for state3 in self.state_space:
            if state3 != state2:
                val_P = self.P_hat.get((state, action, state3), self.default)
                self.P_hat[(state, action, state3)] = val_P + self.alpha_P * (-val_P)

        state_action_pairs = list(itertools.product(self.state_space, self.action_space))
        sampled_pairs = random.sample(state_action_pairs, self.k)

        for (s, a) in sampled_pairs:
            val_q = self.q.get((s, a), self.default)
            sum = 0

            for s2 in self.state_space:
                val_p = self.P_hat.get((s, a, s2), self.default)
                val_r = self.R_hat.get((s, a, s2), self.default)
                m = -np.inf
                for a2 in self.action_space:
                    m = max(m, self.q.get((s2, a2), self.default))

                # todo à vérifier (je ne sais pas si les parenthèses sont bien placées dans le slide)
                # avec val_p * (val_r + self.gamma * m) - val_q (comme sur le slide), les valeurs explosent
                sum += val_p * (val_r + self.gamma * m - val_q)

            self.q[(s, a)] = val_q + self.alpha * sum

    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)

        if done:  # todo à vérifier
            m = 0
        else:
            m = -np.inf
            for action2 in self.action_space:
                m = max(m, self.q.get((state2, action2), self.default))

        val += self.alpha * (reward + self.gamma * m - val)

        self.q[(state, action)] = val

        self._updateMDP(state, action, state2, reward)

