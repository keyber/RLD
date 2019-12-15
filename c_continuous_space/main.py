import matplotlib
matplotlib.use("TkAgg")
import gym
from gym import wrappers
import numpy as np
import A2C
from torch import nn
from torch.optim import Adam
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
import torch
import random


def loop(envm, agent, episode_count=1000, render_period=100, render_sleep=0., verbose=False):
    episode_count = episode_count
    envm.verbose = verbose
    t0 = time()
    
    writer = SummaryWriter()
    l_rsum, l_action_count = [], []

    trajectoire_exemple_en_cours = True
    trajectoire_exemple = []
    
    for i in range(episode_count):
        obs = envm.reset()
        agent.reset(obs)
        
        envm.verbose = (i % render_period == 0 and i > 0)  # afficher 1 episode sur 100
        if envm.verbose:
            print(agent.action_count)
            t0 = time()
            envm.render()
        
        rsum, j = 0, 0
        while True:
            j += 1
            action, state_score, actions_scores = agent.act()
            obs, reward, done, _ = envm.step(action)
            agent.get_result(obs, reward, done)
            
            rsum += reward
            if envm.verbose:
                # print(state_score.item(), actions_scores.data.detach().numpy())
                if trajectoire_exemple_en_cours:
                    trajectoire_exemple.append(obs)
                # print("%.2f"%state_score.item(), (actions_scores.data.detach().numpy()*10).astype(int)/10)
                envm.render()
                sleep(render_sleep)
            if done:
                break
        
        l_rsum.append(rsum)
        l_action_count.append(j)
        writer.add_scalar('reward', rsum, i)
        
        if envm.verbose:
            print(i, "rsum %.2f" % rsum, "mean %.2f"%np.mean(l_rsum), "time", time() - t0, "\n\n")
            l_rsum, l_action_count = [], []
            
            trajectoire_exemple_en_cours = False
            # print("trajectory")
            # for s in trajectoire_exemple:
            #     state_v, actions = agent.get_values(s)
            #     print(state_v.item(), actions.detach().numpy())
            # print()
        

def main_cartPole():
    env = gym.make('CartPole-v1')
    outdir = 'cartpole/A2C'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    
    s = random.randint(0, 2**32) #2935429358
    torch.manual_seed(s)
    np.random.seed(s)
    envm.seed(s)
    print("seed:", s)
    
    sizeIn = env.observation_space.shape[0]     #4  = len(phi(x)) #cart_pos, cart_spe, pole_pos, pos_spe
    sizeOut = env.action_space.n                #[0, 1] # gauche droite
    
    Q = A2C.NN_Q(sizeIn, sizeOut, [24, 24]) #type: nn.Module
    V = A2C.NN_V(sizeIn, 1, [24, 24])       #type: nn.Module
    optim_q = Adam(Q.parameters(), lr=1e-2)
    optim_v = Adam(V.parameters(), lr=1e-2)
    t_max = 5000
    agent = A2C.BatchA2C_Agent(t_max, sizeOut, sizeIn, Q, V, optim_v, optim_q, gamma=.99)
    
    loop(envm, agent, episode_count=1001)
        

    envm.close()


if __name__ == '__main__':
    main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    # main_lunarlander()
