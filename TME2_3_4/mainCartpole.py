import matplotlib

matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
from DQN import DQN_Agent


def identity(x):
    return x

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # Enregistrement de l'Agent
    phi = identity
    sizeIn = 4 #=len(env.state) #=len(phi(x))
    action_space = [0, 1]
    sizeout = len(action_space)
    agent = DQN_Agent(env, action_space, sizeIn, sizeout, C=10, eps=1e-2, replay_memory_max_len=1000, replay_memory_n=100, gamma=1 - 1e-1, phi=identity)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    
    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    lrsum = []

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            print(i//100, "rsum %.2f" % np.mean(lrsum))
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                lrsum.append(rsum)
                # print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()