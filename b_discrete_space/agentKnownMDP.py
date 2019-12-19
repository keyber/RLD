import numpy as np


class RandomAgent:
    """The world's simplest agent!"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self):
        return self.action_space.sample()
    
    def reset(self, obs):
        pass
    
    def get_result(self, obs, reward, done):
        pass
    
    def get_policy(self, states):
        return {s: self.action_space.sample() for s in states}


class PolicyIterationAgent:
    def __init__(self, state_space, action_space, p, gamma=.99):
        self.policy = None
        self.state_values = None
        self.action_space = action_space
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.last_state = None
    
    def get_state_values_from_policy(self, policy, eps=1e-6):
        current_state_values = self.state_values.copy()
        new_state_values = {}
        
        diff = eps
        while diff >= eps:
            
            for state in self.state_space:
                state_value = 0
                
                for proba, state2, reward, done in self.p[state][policy[state]]:
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
            self.state_values = self.get_state_values_from_policy(self.policy, eps)
            
            new_policy = self._update_policy()
            
            change = False
            for state in self.policy.keys():
                if self.policy[state] != new_policy[state]:
                    change = True
                    break
            
            self.policy = new_policy
    
    def act(self):
        return self.policy[self.last_state]
    
    def reset(self, obs):
        self.last_state = obs
    
    def get_result(self, obs, _reward, _done):
        self.last_state = obs
    
    def evaluate_other_agent_policy(self, agent, eps=1e-3):
        p = agent.get_policy(self.state_space)
        u = self.get_state_values_from_policy(p, eps=eps)
        v = self.state_values
        
        assert u.keys() == v.keys()
        
        d = np.array([v[k] - u[k] for k in u])
        assert np.all(d > -1e-3)
        
        return np.sum(d)
    
    def get_policy(self, _states):
        return self.policy


class ValueIterationAgent:
    def __init__(self, state_space, _action_space, p, gamma=.99):
        self.policy = None
        self.state_values = None
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.last_state = None
    
    def _value_iteration(self, eps):
        current_state_values = self.state_values
        new_state_values = {}
        diff = eps
        
        while diff >= eps:
            for state in self.state_space:
                state_value = -np.inf
                
                for action in self.p[state].keys():
                    value = 0
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
        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in self.state_space}
        
        self.state_values = self._value_iteration(eps)
        
        self.policy = self._update_policy()
    
    def act(self):
        return self.policy[self.last_state]
    
    def reset(self, obs):
        self.last_state = obs
    
    def get_result(self, obs, _reward, _done):
        self.last_state = obs
