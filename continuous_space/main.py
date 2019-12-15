import matplotlib
matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
import agentA2C
from torch import nn
from torch.optim import Adam
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter


def loop(env, agent, outdir):
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    
    episode_count = 1000
    env.verbose = True
    np.random.seed(5)
    t0 = time()
    
    writer = SummaryWriter()
    l_rsum, l_action_count = [], []

    for i in range(episode_count):
        obs = envm.reset()
        agent.reset(obs)
        
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            print(agent.action_count)
            t0 = time()
            env.render()
        
        rsum, j = 0, 0
        while True:
            j += 1
            action, state_score, actions_scores = agent.act()
            obs, reward, done, _ = envm.step(action)
            losses = agent.get_result(obs, reward, done)
            
            if losses is not None:
                writer.add_scalars('loss', losses, i)
            
            rsum += reward
            if env.verbose:
                print("%.2f"%state_score.item(), (actions_scores.data.detach().numpy()*10).astype(int)/10)
                env.render()
                sleep(1.)
            if done:
                break
        
        l_rsum.append(rsum)
        l_action_count.append(j)
        writer.add_scalar('reward', rsum, i)
        
        if env.verbose:
            print(i, "rsum %.2f" % rsum, "mean %.2f"%np.mean(l_rsum), "time", time() - t0, "\n\n")
            l_rsum, l_action_count = [], []
        
    print("done")
    env.close()

def main_cartPole():
    env = gym.make('CartPole-v1')
    env.render()
    sizeIn = env.observation_space.shape[0]     #4  = len(phi(x)) #cart_pos, cart_spe, pole_pos, pos_spe
    sizeOut = env.action_space.n                #[0, 1] # gauche droite
    Q = agentA2C.NN_Q(sizeIn, sizeOut, [24, 24]) #type: nn.Module
    V = agentA2C.NN_V(sizeIn, 1, [24, 24])       #type: nn.Module
    optim_q = Adam(Q.parameters(), lr=1e-2)
    optim_v = Adam(V.parameters(), lr=1e-2)
    t_max = 5000
    agent = agentA2C.BatchA2C_Agent(t_max, sizeOut, sizeIn, Q, V, optim_v, optim_q, gamma=.99)
    
    
    outdir = 'cartpole/A2C'
    loop(env, agent, outdir)


if __name__ == '__main__':
    main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    # main_lunarlander()
