import numpy as np

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
