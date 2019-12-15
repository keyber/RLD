import gym
from gym import wrappers
from b_discrete_space.agentKnownMDP import ValueIterationAgent, PolicyIterationAgent, RandomAgent
from b_discrete_space.qlearning import SarsaAgent, QLearningAgent, DynaQAgent
#from A2C import BatchA3C_Agent
import matplotlib
matplotlib.use("TkAgg")
# noinspection PyUnresolvedReferences
import gridworld # import non utilisé ensuite mais nécessaire
import numpy as np
import matplotlib.pyplot as plt
from time import time

EPISODE_COUNT = 1000

def _main_demo(env, agent, name, plan_id):
    # env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  #visualisation sur la console

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0-{}/{}-results'.format(plan_id, name)
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    episode_count = 1000
    reward = 0
    done = False
    FPS = 1e-6  # ~temps de pause entre deux affichages
    
    all_rsums = []

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
                all_rsums.append(rsum)
                break
    
    print("done")
    print("Average rsum : {} +/- {}".format(np.mean(all_rsums), np.std(all_rsums)))
    env.close()

def get_learning_curve(env, agent, name, plan_id):
    env.render(mode="human")  # visualisation sur la console
    outdir = 'gridworld-v0-{}/{}-results'.format(plan_id, name)

    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    episode_count = EPISODE_COUNT
    reward = 0
    done = False

    all_rewards = []

    print("Running {} :".format(name))

    for i in range(episode_count):
        obs = envm.reset()
        rsum = 0
        j = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if done:
                if i % 100 == 0:
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        all_rewards.append(rsum)

    env.close()
    return np.cumsum(all_rewards)

def _main_learning_curve():
    env = gym.make("gridworld-v0")
    plan_id = "10"
    env.seed(0)  # Initialise le seed du pseudo-random
    env.setPlan("gridworldPlans/plan{}.txt".format(plan_id), {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    state, transitions = list(mdp.items())[0]

    agent_random = RandomAgent(env.action_space)
    agent_qlearning = QLearningAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1)
    agent_sarsa = SarsaAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1, cst=-.01)
    agent_dynaq = DynaQAgent(env, list(mdp[state].keys()), mdp.keys(), default=0, alpha=.5, gamma=1 - 1e-1, eps=.1,
                        alpha_R=0.5, alpha_P=0.5, k=5)

    learning_curve_random = get_learning_curve(env, agent_random, "random-agent", plan_id)
    learning_curve_qlearning = get_learning_curve(env, agent_qlearning, "qlearning-agent", plan_id)
    learning_curve_sarsa = get_learning_curve(env, agent_sarsa, "sarsa-agent", plan_id)
    learning_curve_dynaq = get_learning_curve(env, agent_dynaq, "dynaq-agent", plan_id)

    plt.plot(learning_curve_random, label="random", color="black")
    plt.plot(learning_curve_qlearning, label="qlearning", color="blue")
    plt.plot(learning_curve_sarsa, label="sarsa", color="green")
    plt.plot(learning_curve_dynaq, label="dyna-Q", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

    plt.plot(learning_curve_qlearning, label="qlearning", color="blue")
    plt.plot(learning_curve_sarsa, label="sarsa", color="green")
    plt.plot(learning_curve_dynaq, label="dyna-Q", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

def _main_perf():
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
    print("timeP moy:", sum(timeP) / len(timeP), "max:", max(timeP))
    print("timeV moy:", sum(timeV) / len(timeV), "max:", max(timeV))
    
    env.close()

def identity(x):
    return x

def main():
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("gridworld-v0")
    # agent_name = "random-agent"
    # agent_name = "policy-agent"
    # agent_name = "value-agent"
    plan_id = "0"

    env.seed(0)  # Initialise le seed du pseudo-random
    # env.setPlan("gridworldPlans/plan{}.txt".format(plan_id), {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    # print("actions possibles:", env.action_space)  # Quelles sont les actions possibles
    # print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    # print("Nombre d'etats:", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(env.action_space)

    # print("ex state:", state)  # un etat du mdp
    # print("ex transistions:",
    #      transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    
    # Execution avec un Agent
    # agent = RandomAgent(env.action_space)
    # agent = PolicyIterationAgent(env, mdp.keys(), env.action_space,
    #                            mdp, env.getMDP()[0], gamma=1-1e-3, cst=-.01)
    # agent = ValueIterationAgent(env, mdp.keys(), env.action_space,
    #                              mdp, env.getMDP()[0], gamma=1 - 1e-3, cst=-.01)

    # agent.compute_best_policy(eps=1e-15)

    # agent_random = RandomAgent(env.action_space)
    # agent_qlearning = QLearningAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1)
    # agent_sarsa = SarsaAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1-1e-1, eps=.1, cst=-.01)
    # agent_dynaq = DynaQAgent(env, list(mdp[state].keys()), mdp.keys(), default=0, alpha=.5, gamma=1 - 1e-1, eps=.1,
    #                     alpha_R=0.5, alpha_P=0.5, k=5)

    # phi = identity
    # sizeIn = 4 #=len(env.state) #=len(phi(x))
    # action_space = [0, 1]
    # sizeout = len(action_space)
    #agent = DQN_Agent(env, action_space, sizeIn, sizeout, T=10, C=10, eps=1e-2, replay_memory_max_len=10, replay_memory_n=8, gamma=1 - 1e-1, phi=identity)
    #agent = BatchA3C_Agent(env, action_space, sizeIn, sizeout, T=10, C=10, eps=1e-2, replay_memory_max_len=10, replay_memory_n=8, gamma=1 - 1e-1, phi=identity)

    ##### CONTINU #############
    #state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0]
    #max_action = float(env.action_space.high[0])

    #agent = DDPG(env, state_dim, action_dim, max_action)

    #_main_demo(env, agent, agent_name, plan_id)
    # _main_perf()


if __name__ == '__main__':
    main()
