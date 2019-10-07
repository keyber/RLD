from rl import SarsaAgent, QLearningAgent, ValueIterationAgent, DynaQAgent, PolicyIterationAgent
import gym
from gym import wrappers

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
    
    agent = SarsaAgent(env, list(mdp[state].keys()), default=0, alpha=.5, gamma=1 - 1e-1, eps=.1, cst=-.01)
    
    _main_demo(env, agent)
    
    # _main_perf()


if __name__ == '__main__':
    main()

