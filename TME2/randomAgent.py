import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class PolicyIterationAgent():
    def __init__(self, env, state_space, action_space, p, obs_to_states, gamma=.99):
        # initialisation de la politique aléatoirement
        self.policy = {state: action_space.sample() for state in state_space}

        # initialisation des valeurs aléatoirement
        self.state_values = {state: 0 for state in state_space}
        self.action_space = action_space
        self.state_space = state_space
        self.p = p
        self.gamma = gamma
        self.obs_to_states = obs_to_states
        self.env = env
    
    def get_state_values_from_policy(self, eps=1e-3):
        current_state_values = self.state_values
        new_state_values = {}

        diff = eps
        while diff >= eps:
            diff = 0

            for state in self.state_space:
                s = 0
                for proba, state2, reward, done in self.p[state][self.policy[state]]:
                    val = reward
                    if not done:
                        val += self.gamma * current_state_values[state2]
                    val *= proba

                    s +=  val
                
                diff += (s - current_state_values[state]) ** 2
                new_state_values[state] = s
            
            diff = np.sqrt(diff)
            current_state_values = new_state_values

        return new_state_values
    
    def update_policy(self):
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

    def compute_best_policy(self):
        change = True
        while change:
            # on ne réinitialise pas V à chaque itération
            self.state_values = self.get_state_values_from_policy()

            new_policy = self.update_policy()

            change = False
            for state in self.policy.keys():
                if self.policy[state] != new_policy[state]:
                    change = True
                    break
            
            self.policy = new_policy

    def act(self, observation, reward, done):
        current_state = self.env.state2str(observation)
        return self.policy[current_state]


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    #agent = RandomAgent(env.action_space)
    agent = PolicyIterationAgent(env, env.getMDP()[0].values(), env.action_space,
                                env.getMDP()[1], env.getMDP()[0], gamma=.99)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
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