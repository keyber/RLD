import itertools
import random

import matplotlib

matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
# noinspection PyUnresolvedReferences
import gridworld # import non utilisé ensuite mais nécessaire


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, _observation, _reward, _done):
        return self.action_space.sample()


class PolicyIterationAgent:
    def __init__(self, env, state_space, action_space, p, obs_to_states, gamma=.99, cst=-.1):
        self.policy = None
        self.state_values = None
        self.action_space = action_space
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.cst = cst
        self.obs_to_states = obs_to_states
        self.env = env
    
    def _get_state_values_from_policy(self, eps):
        current_state_values = self.state_values
        new_state_values = {}
        
        diff = eps
        while diff >= eps:
            
            for state in self.state_space:
                state_value = self.cst
                
                for proba, state2, reward, done in self.p[state][self.policy[state]]:
                    val = reward
                    if not done:
                        val += self.gamma * current_state_values[state2]
                    val *= proba
                    
                    state_value += val
                
                new_state_values[state] = state_value
                assert current_state_values is not new_state_values
            
            diff = 0
            for state in self.state_space:
                diff += (new_state_values[state] - current_state_values[state]) ** 2
                current_state_values[state] = new_state_values[state]
            diff = np.sqrt(diff)
        
        return new_state_values
    
    def _update_policy(self):
        new_policy = {}
        for state in self.state_space:
            best_action = None
            max_value = -np.inf
            
            for action in self.p[state].keys():
                s = 0
                for proba, state2, reward, done in self.p[state][action]:
                    val = reward
                    if not done:
                        val += self.gamma * self.state_values[state2]
                    val *= proba
                    s += val
                
                if s > max_value:
                    best_action = action
                    max_value = s
            
            new_policy[state] = best_action
        
        return new_policy
    
    def compute_best_policy(self, eps=1e-6):
        # initialisation de la politique aléatoirement
        self.policy = {state: self.action_space.sample() for state in self.state_space}
    
        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in self.state_space}
    
        change = True
        while change:
            # on ne réinitialise pas V à chaque itération
            self.state_values = self._get_state_values_from_policy(eps)
            
            new_policy = self._update_policy()
            
            change = False
            for state in self.policy.keys():
                if self.policy[state] != new_policy[state]:
                    change = True
                    break
            
            self.policy = new_policy
    
    def act(self, observation, _reward, _done):
        current_state = self.env.state2str(observation)
        return self.policy[current_state]


class ValueIterationAgent:
    def __init__(self, env, state_space, _action_space, p, _obs_to_states, gamma=.99, cst=-.1):
        self.policy = None
        self.state_values = None
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.cst = cst
        self.env = env
    
    def _value_iteration(self, eps):
        current_state_values = self.state_values
        new_state_values = {}
        diff = eps
    
        while diff >= eps:
            for state in self.state_space:
                state_value = -np.inf
            
                for action in self.p[state].keys():
                    value = self.cst
                    for proba, state2, reward, done in self.p[state][action]:
                        val = reward
                        if not done:
                            val += self.gamma * current_state_values[state2]
                        val *= proba
                    
                        value += val
                
                    state_value = max(state_value, value)
            
                new_state_values[state] = state_value
        
            diff = 0
            for state in self.state_space:
                diff += (new_state_values[state] - current_state_values[state]) ** 2
                current_state_values[state] = new_state_values[state]
            diff = np.sqrt(diff)
            
        return new_state_values

    def _update_policy(self):
        """meme code que policy_iteration"""
        new_policy = {}
        for state in self.state_space:
            best_action = None
            max_value = -np.inf
        
            for action in self.p[state].keys():
                s = self.cst
                for proba, state2, reward, done in self.p[state][action]:
                    val = reward
                    if not done:
                        val += self.gamma * self.state_values[state2]
                    val *= proba
                    s += val
            
                if s > max_value:
                    best_action = action
                    max_value = s
        
            new_policy[state] = best_action
    
        return new_policy

    def compute_best_policy(self, eps=1e-6):
        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in self.state_space}
        
        self.state_values = self._value_iteration(eps)
        
        self.policy = self._update_policy()
        
    
    def act(self, observation, _reward, _done):
        current_state = self.env.state2str(observation)
        return self.policy[current_state]


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

