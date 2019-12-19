import gym
from agentKnownMDP import ValueIterationAgent, PolicyIterationAgent, RandomAgent
from qlearning import SarsaAgent, QLearningAgent, DynaQAgent
import matplotlib
matplotlib.use("TkAgg")
# noinspection PyUnresolvedReferences
import gridworld  # import non utilisé ensuite mais nécessaire
import numpy as np
import matplotlib.pyplot as plt
from time import time

# l'environnement 9 cause un MaximumRecursionDeptError (même avec une limite à 10000)
available_envs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]


def get_env(plan_id, default_reward=-.001, seed=0):
    env = gym.make("gridworld-v0")
    
    rewards = {0: default_reward, 3: 1, 4: 1, 5: -1, 6: -1}
    env.setPlan("gridworldPlans/plan{}.txt".format(plan_id), rewards)
    
    env.seed(seed)
    return env


def _main_print():
    env = get_env(0)
    
    statedic, mdp = env.getMDP()
    
    # statedic : etat-> numero de l'etat
    print("Nombre d'etats:", len(statedic))
    assert len(statedic) == env.observation_space.n
    
    state, transitions = next(iter(mdp.items()))  # récupère un etat du mdp
    
    # mdp.keys() ne contient pas les états terminaux
    # env.observation_space les contient
    print(env.action_space)  # Discrete <=> int
    print(list(mdp[state].keys()))
    assert len(list(mdp[state].keys())) != env.observation_space.n
    
    print("ex state:", state)
    
    # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    print("ex transistions:", transitions)
    
    for i in available_envs:
        env = get_env(i)
        env.getMDP()
        print(env.observation_space)


def _main_perf(plan_ids):
    list_cst = [0, -.001, -.1]
    list_gamma = [.9, .999]
    n = 1
    
    for plan_id in plan_ids:
        print("plan_id", plan_id)
        timeP = []
        timeV = []
        for cst in list_cst:
            for gamma in list_gamma:
                env = get_env(plan_id, default_reward=cst)
                statedic, mdp = env.getMDP()
                
                for _ in range(n):
                    t0 = time()
                    agentP = PolicyIterationAgent(mdp.keys(), env.action_space, mdp, gamma=gamma)
                    agentP.compute_best_policy(eps=1e-6)
                    timeP.append(time() - t0)
                    
                    t0 = time()
                    agentV = ValueIterationAgent(mdp.keys(), env.action_space, mdp, gamma=gamma)
                    agentV.compute_best_policy(eps=1e-6)
                    timeV.append(time() - t0)
                    
                    assert agentP.policy.keys() == agentV.policy.keys()
                    for k in agentP.policy.keys():
                        # même politique ou actions de même valeur
                        assert agentP.policy[k] == agentV.policy[k] \
                               or abs(agentP.state_values[k] - agentV.state_values[k]) < 1e-5
                
                env.close()
        
        print("timeP moy:", sum(timeP) / len(timeP), "max:", max(timeP))
        print("timeV moy:", sum(timeV) / len(timeV), "max:", max(timeV))


def _main_demo(plan_id=0):
    env = get_env(plan_id)
    mdp = env.getMDP()[1]
    state, transitions = next(iter(mdp.items()))
    action_space = list(mdp[state].keys())
    
    agent = QLearningAgent(action_space)
    
    # env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  #visualisation sur la console
    
    # Faire un fichier de log sur plusieurs scenarios
    episode_count = 1000
    FPS = 1e-6  # ~temps de pause entre deux affichages
    
    all_rsums = []
    
    for i in range(episode_count):
        obs = env.state2str(env.reset())
        agent.reset(obs)
        env.verbose = (i % 10 == 0 and i > 0)  # afficher 1 episode sur 10
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            obs = env.state2str(obs)
            agent.get_result(obs, reward, done)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                all_rsums.append(rsum)
                break
    
    print("done")
    print("Average rsum : {} +/- {}".format(np.mean(all_rsums), np.std(all_rsums)))
    env.close()


def get_learning_curve(env, agent, name, evaluator, n_iter):
    env.render(mode="human")  # visualisation sur la console
    episode_count = n_iter
    
    all_rewards = []
    all_actions = []
    all_policy_values = []
    
    print("Running {} :".format(name))
    
    for i in range(episode_count):
        obs = env.state2str(env.reset())
        agent.test_mode = (i % 10 == 0)
        agent.reset(obs)
        rsum = 0
        j = 0
        while True:
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            obs = env.state2str(obs)
            agent.get_result(obs, reward, done)
            
            rsum += reward
            j += 1
            if done:
                if i % 100 == 0:
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        if agent.test_mode:
            all_rewards.append(rsum)
            all_actions.append(j)
            all_policy_values.append(evaluator.evaluate_other_agent_policy(agent))
    
    env.close()
    
    w = 10  # moving average window width
    all_actions = np.convolve(all_actions, np.ones(w), 'valid') / w
    # all_policy_values = np.convolve(all_policy_values, np.ones(w), 'valid') / w
    
    return all_policy_values, np.cumsum(all_rewards), all_actions


# noinspection PyUnresolvedReferences
def _main_learning_curve(plan_ids, n_iter):
    for plan_id in plan_ids:
        env = get_env(plan_id, default_reward=-0.001)
        mdp = env.getMDP()[1]
        non_terminal_states = mdp.keys()
        state, transitions = next(iter(mdp.items()))
        action_space = list(mdp[state].keys())
        
        evaluator = PolicyIterationAgent(mdp.keys(), env.action_space, mdp)
        evaluator.compute_best_policy()
        
        agents = [
            ("Optimal", "black", evaluator),
            ("Random", "gray", RandomAgent(env.action_space)),
            ("QLearning", "red", QLearningAgent(action_space)),
            ("Sarsa", "blue", SarsaAgent(action_space)),
            ("Dyna-Q", "green", DynaQAgent(action_space, non_terminal_states, lr_R=0.5, lr_P=0.5, k=5))
        ]
        
        
        res = [(name, color, get_learning_curve(env, agent, name=name, evaluator=evaluator, n_iter=n_iter))
               for name, color, agent in agents]

        plt.suptitle("map " + str(plan_id), fontsize=16)
        plt.title("Policy loss")
        for name, color, (policy_values, cum_reward, action_count) in res:
            if name == "Optimal":
                plt.plot(policy_values, "--", label=name, color=color)
            else:
                plt.plot(policy_values, label=name, color=color)
        plt.legend()
        
        plt.figure()
        plt.suptitle("map " + str(plan_id), fontsize=16)
        plt.title("Cumulated reward")
        for name, color, (policy_values, cum_reward, action_count) in res:
            if name == "Optimal":
                plt.plot(cum_reward, "--", label=name, color=color)
            else:
                plt.plot(cum_reward, label=name, color=color)
        plt.legend()
        plt.show()


def main():
    # utilisation de l'API
    # _main_print()
    
    # mesure le temps de convergence de Value et Policy Iteration
    # _main_perf(available_envs)
    
    # plot les learning curves de DQN, Sarsa et DynaQ
    _main_learning_curve(available_envs[:1], n_iter=1000)
    
    # montre un agent jouer
    _main_demo()


main()
