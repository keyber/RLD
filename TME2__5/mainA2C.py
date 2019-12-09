import matplotlib
matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
from A2C import BatchA2C_Agent, NN_Q, NN_V
from torch.optim import Adam
from time import time
from torch.utils.tensorboard import SummaryWriter


def loop(env, agent, outdir):
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    
    episode_count = 1000000
    reward = None
    done = False
    env.verbose = True
    np.random.seed(5)
    lrsum = []
    t0 = time()
    
    writer = SummaryWriter()

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 10 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            print(i, "rsum %.2f" % np.mean(lrsum), "action_count", "time", time() - t0)
            writer.add_scalar('reward', np.mean(lrsum), i)
            t0 = time()
            lrsum = []
            env.render()
        
        rsum = 0
        while True:
            action, losses = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            
            if losses is not None:
                writer.add_scalars('loss', losses, i)
            
            rsum += reward
            if env.verbose:
                env.render()
            if done:
                lrsum.append(rsum)
                # print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()

def main_cartPole():
    env = gym.make('CartPole-v1')
    # Enregistrement de l'Agent

    sizeIn = env.observation_space.shape[0]     #4  = len(phi(x)) #cart_pos, cart_spe, pole_pos, pos_spe
    sizeOut = env.action_space.n                #[0, 1] # gauche droite
    Q = NN_Q(sizeIn, sizeOut, [24, 24])
    V = NN_V(sizeIn, 1, [24, 24])
    optim_q = Adam(Q.parameters(), lr=1e-4)
    optim_v = Adam(V.parameters(), lr=1e-2)
    t_max = 50
    
    agent = BatchA2C_Agent(t_max, env, sizeOut, sizeIn, Q, V, optim_v, optim_q, gamma=.99)

    outdir = 'cartpole-v0/random-agent-results'
    loop(env, agent, outdir)


if __name__ == '__main__':
    main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    # main_lunarlander()
