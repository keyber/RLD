import matplotlib

matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
from DQN import NN, DQN_Agent
from torch.optim import Adam
from time import time
import torch.nn
import matplotlib.pyplot as plt

def identity(x):
    return x

def loop(env, agent, outdir, render=False, feat_extractor=None):
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    
    episode_count = 1000
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


if __name__ == '__main__':
    main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    # main_lunarlander()
