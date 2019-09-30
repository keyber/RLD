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
    def __init__(self, env, action_space, default, alpha, gamma, eps, cst):
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
        super().__init__(env, action_space, default, alpha, gamma, eps, cst)
    
    def _updateQ(self, state, action, state2, reward, done):
        val = self.q.get((state, action), self.default)
        
        if done:  # todo à vérifier
            val += self.alpha * reward
        else:
            action2 = self.epsilon_greedy(state, self.eps)
            val += self.alpha * (reward + self.gamma * self.q.get((state2, action2), self.default) - val)
        
        self.q[(state, action)] = val
    

class DynaQAgent:
    pass

def _main_demo(env, agent):
    # env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  #visualisation sur la console
    
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    FPS = 1e-6  # ~temps de pause entre deux affichages
    assert FPS > 0
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
    
    print("done")
    env.close()

def _main_perf():
    from time import time
    env = gym.make("gridworld-v0")
    env.seed()
    statedic, mdp = env.getMDP()
    timeP = []
    timeV = []
    n = 3
    list_gamma = [1 - 1e-1, 1 - 1e-3]
    list_cst = [0, 1e-3, 1e-1]
    for gamma in list_gamma:
        for cst in list_cst:
            for _ in range(3):
                t0 = time()
                agentP = PolicyIterationAgent(env, mdp.keys(), env.action_space,
                                             mdp, env.getMDP()[0], gamma=gamma, cst=cst)
                agentP.compute_best_policy(eps=1e-15)
                timeP.append(time() - t0)
                
                t0 = time()
                agentV = ValueIterationAgent(env, mdp.keys(), env.action_space,
                                            mdp, env.getMDP()[0], gamma=gamma, cst=cst)
                agentV.compute_best_policy(eps=1e-15)
                timeV.append(time() - t0)
                
                assert agentP.policy.keys() == agentV.policy.keys()
                for k in agentP.policy.keys():
                    assert agentP.policy[k] == agentV.policy[k]
    
    print("tests passés")
    print("timeP moy:", sum(timeP)/len(timeP), "max:", max(timeP))
    print("timeV moy:", sum(timeV)/len(timeV), "max:", max(timeV))
    
    env.close()

def _main_qlearning():
    pass
    
def main():
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print("actions possibles:", env.action_space)  # Quelles sont les actions possibles
    # print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats:", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print("ex state:", state)  # un etat du mdp
    print("ex transistions:",
          transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    
    # Execution avec un Agent
    # agent = PolicyIterationAgent(env, mdp.keys(), env.action_space,
    #                             mdp, env.getMDP()[0], gamma=1-1e-3, cst=-.01)
    # agent = ValueIterationAgent(env, mdp.keys(), env.action_space,
    #                             mdp, env.getMDP()[0], gamma=1 - 1e-3, cst=-.01)
    
    # agent = QLearningAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1, cst=-.01)
    
    agent = SarsaAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1, cst=-.01)
    
    _main_demo(env, agent)
    
    # _main_perf()


if __name__ == '__main__':
    main()

