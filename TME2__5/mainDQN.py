import matplotlib

matplotlib.use("TkAgg")
import gym
from gym import wrappers
import gridworld # import non utilisé ensuite mais nécessaire
import numpy as np
from DQN import NN, DQN_Agent
from torch.optim import Adam#, RMSprop
from time import time
import torch.nn
import ast
import matplotlib.pyplot as plt

class FeaturesExtractor(object):
    def __init__(self, outSize):
        super().__init__()
        self.outSize = outSize*3

    def getFeatures(self, obs):
        '''
            prend en argument la carte du jeu originale et retourne un
            vecteur de features contenant la position de l'agent, celles des éléments jaunes ainsi
            que celles des éléments roses
        '''
        state = np.zeros((3, np.shape(obs)[0], np.shape(obs)[1]))
        state[0] = np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.flatten()

def identity(x):
    return x

def loop(env, agent, outdir, render=False, feat_extractor=None):
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    
    episode_count = 1000 # 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    lrsum = []
    t0 = time()

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            print(i, "rsum %.2f" % np.mean(lrsum), "action_count", "time", time() - t0)
            t0 = time()
            if render:
                env.render()
        j = 0
        rsum = 0
        while True:
            if feat_extractor:
                obs = feat_extractor.getFeatures(obs)

            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose and render:
                env.render()
            if done:
                lrsum.append(rsum)
                # print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    return lrsum


def main_cartPole():
    env = gym.make('CartPole-v1')
    # Enregistrement de l'Agent

    sizeIn = env.observation_space.shape[0]     #4  = len(phi(x)) #cart_pos, cart_spe, pole_pos, pos_spe
    sizeOut = env.action_space.n                #[0, 1] # gauche droite
    Q = NN(sizeIn, sizeOut, [24, 24])
    optim = Adam(Q.parameters(), lr=1e-3)
    agent = DQN_Agent(env, range(env.action_space.n),
                       Q=Q,
                       optim=optim,
                       loss=torch.nn.MSELoss(),
                       C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                       replay_memory_n=100, gamma=1 - 1e-1, phi=identity)
    agent_noExpReplay = DQN_Agent(env, range(env.action_space.n),
                      Q=Q,
                      optim=optim,
                      loss=torch.nn.MSELoss(),
                      C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                      replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_exp_replay=False)
    agent_noTargetNetwork = DQN_Agent(env, range(env.action_space.n),
                                  Q=Q,
                                  optim=optim,
                                  loss=torch.nn.MSELoss(),
                                  C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                                  replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_target_network=False)
    outdir = 'cartpole-v1/DQN-results'
    outdir_noExpReplay = 'cartpole-v1/DQN-noExpReplay-results'
    outdir_noTargetNetwork = 'cartpole-v1/DQN-noTargetNetwork-results'

    lrsum = loop(env, agent, outdir, render=False)
    lrsum_noExpReplay = loop(env, agent_noExpReplay, outdir_noExpReplay, render=False)
    lrsum_noTargetNetwork = loop(env, agent_noTargetNetwork, outdir_noTargetNetwork, render=False)

    plt.plot(np.cumsum(lrsum), label="DQN", color="blue")
    plt.plot(np.cumsum(lrsum_noExpReplay), label="DQN-noExpReplay", color="red")
    plt.plot(np.cumsum(lrsum_noTargetNetwork), label="DQN-noTargetNetwork", color="green")
    plt.legend()
    plt.show()

def main_acrobot():
    env = gym.make('Acrobot-v1')
    # Enregistrement de l'Agent

    sizeIn = env.observation_space.shape[0]
    sizeOut = env.action_space.n
    Q = NN(sizeIn, sizeOut, [])
    optim = Adam(Q.parameters(), lr=1e-3)
    agent = DQN_Agent(env, range(env.action_space.n),
                       Q=Q,
                       optim=optim,
                       loss=torch.nn.MSELoss(),
                       C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                       replay_memory_n=100, gamma=1 - 1e-1, phi=identity)
    
    outdir = 'Acrobot-v1/random-agent-results'
    loop(env, agent, outdir)

def main_mountainCar():
    env = gym.make('MountainCar-v0')
    # Enregistrement de l'Agent

    sizeIn = env.observation_space.shape[0]
    sizeOut = env.action_space.n
    Q = NN(sizeIn, sizeOut, [])
    optim = Adam(Q.parameters(), lr=1e-3)
    agent = DQN_Agent(env, range(env.action_space.n),
                       Q=Q,
                       optim=optim,
                       loss=torch.nn.MSELoss(),
                       C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                       replay_memory_n=100, gamma=1 - 1e-1, phi=identity)

    outdir = 'MountainCar-v0/random-agent-results'
    loop(env, agent, outdir)

def main_lunarlander():
    env = gym.make('LunarLander-v2')
    # Enregistrement de l'Agent

    sizeIn = env.observation_space.shape[0]
    sizeOut = env.action_space.n
    Q = NN(sizeIn, sizeOut, [24, 24])
    optim = Adam(Q.parameters(), lr=1e-3)
    agent = DQN_Agent(env, range(env.action_space.n),
                       Q=Q,
                       optim=optim,
                       loss=torch.nn.MSELoss(),
                       C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                       replay_memory_n=100, gamma=1 - 1e-1, phi=identity)
    agent_noExpReplay = DQN_Agent(env, range(env.action_space.n),
                      Q=Q,
                      optim=optim,
                      loss=torch.nn.MSELoss(),
                      C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                      replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_exp_replay=False)
    agent_noTargetNetwork = DQN_Agent(env, range(env.action_space.n),
                      Q=Q,
                      optim=optim,
                      loss=torch.nn.MSELoss(),
                      C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                      replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_target_network=False)
    outdir = 'LunarLander-v2/DQN-results'
    outdir_noExpReplay = 'LunarLander-v2/DQN-noExpReplay-results'
    outdir_noTargetNetwork = 'LunarLander-v2/DQN-noTargetNetwork-results'

    lrsum = loop(env, agent, outdir, render=False)
    lrsum_noExpReplay = loop(env, agent_noExpReplay, outdir_noExpReplay, render=False)
    lrsum_noTargetNetwork = loop(env, agent_noTargetNetwork, outdir_noTargetNetwork, render=False)

    plt.plot(np.cumsum(lrsum), label="DQN", color="blue")
    plt.plot(np.cumsum(lrsum_noExpReplay), label="DQN-noExpReplay", color="red")
    plt.plot(np.cumsum(lrsum_noTargetNetwork), label="DQN-noTargetNetwork", color="green")
    plt.legend()
    plt.show()


def main_gridWorld():
    env = gym.make("gridworld-v0")

    sizeIn = 3 * 6 * 6
    sizeOut = env.action_space.n
    Q = NN(sizeIn, sizeOut, [24, 24])
    optim = Adam(Q.parameters(), lr=1e-3)
    agent = DQN_Agent(env, range(env.action_space.n),
                      Q=Q,
                      optim=optim,
                      loss=torch.nn.MSELoss(),
                      C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                      replay_memory_n=100, gamma=1 - 1e-1, phi=identity)
    agent_noExpReplay = DQN_Agent(env, range(env.action_space.n),
                      Q=Q,
                      optim=optim,
                      loss=torch.nn.MSELoss(),
                      C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                      replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_exp_replay=False)
    agent_noTargetNetwork = DQN_Agent(env, range(env.action_space.n),
                                  Q=Q,
                                  optim=optim,
                                  loss=torch.nn.MSELoss(),
                                  C=10, eps=1.0, eps_decay=.99, replay_memory_max_len=1000,
                                  replay_memory_n=100, gamma=1 - 1e-1, phi=identity, with_target_network=False)

    extractor = FeaturesExtractor(sizeOut)

    outdir = 'GridWorld-v0/DQN-results'
    outdir_noExpReplay = 'GridWorld-v0/DQN-noTargetNetwork-results'
    outdir_noTargetNetwork = 'GridWorld-v0/DQN-noTargetNetwork-results'

    lrsum = loop(env, agent, outdir, render=False, feat_extractor=extractor)
    lrsum_noExpReplay = loop(env, agent_noExpReplay, outdir_noExpReplay, render=False, feat_extractor=extractor)
    lrsum_noTargetNetwork = loop(env, agent_noTargetNetwork, outdir_noTargetNetwork, render=False, feat_extractor=extractor)

    plt.plot(np.cumsum(lrsum), label="DQN", color="blue")
    plt.plot(np.cumsum(lrsum_noExpReplay), label="DQN-noExpReplay", color="red")
    plt.plot(np.cumsum(lrsum_noTargetNetwork), label="DQN-noTargetNetwork", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    main_lunarlander()
    # main_gridWorld()
