import matplotlib
matplotlib.use("TkAgg")
import gym
from A2C import *
from torch import nn
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
import torch
import random

import utils

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def loop(env, agent, writer, episode_count=1000, render_period=10, render_sleep=.1, verbose=0):
    episode_count = episode_count
    t0 = time()
    space_bounds = np.array([[np.inf] * agent.state_space, [-np.inf] * agent.state_space]).T

    l_rsum, l_action_count = [], []

    debug_trajectories = []
    if verbose:
        print("debug trajectories")
        for a in range(agent.action_space):
            l = []
            env.reset()
            done = False
            while not done:
                obs, _, done, _ = env.step(a)
                l.append(obs)
            debug_trajectories.append(l)
            print("action", a, "trajectory length", len(l))
        
        agent.benchmarked_trajectories = debug_trajectories
        agent.space_bounds = space_bounds
        print()
    
    for i in range(episode_count):
        obs = env.reset()
        agent.reset(obs)
        
        env.verbose = (i % render_period == 0 and i > 0)  # affiche 1 episode sur 100
        if env.verbose:
            print(agent.action_count)
            env.render()
        
        rsum, j = 0, 0
        while True:
            j += 1
            action = agent.act()
            obs, reward, done, _ = env.step(action)
            space_bounds[:, 0] = np.minimum(space_bounds[:, 0], obs)
            space_bounds[:, 1] = np.maximum(space_bounds[:, 1], obs)
            agent.get_result(obs, reward, done)
            
            rsum += reward
            if env.verbose:
                # print(state_score.item(), actions_scores.data.detach().numpy())
                env.render()
                sleep(render_sleep)
            if done:
                break
        
        l_rsum.append(rsum)
        l_action_count.append(j)
        writer.add_scalar('reward', rsum, i)
        if verbose:
            print(i, rsum)
        
        if env.verbose:
            print(i, "rsum %.2f" % rsum, "mean %.2f"%np.mean(l_rsum), "time", time() - t0, "\n\n")
            t0 = time()
            l_rsum, l_action_count = [], []
        

def main_cartPole():
    env = gym.make('CartPole-v1')
    writer = SummaryWriter()
    
    s = random.randint(0, 2**32)
    torch.manual_seed(s)
    np.random.seed(s)
    env.seed(s)
    print("seed:", s)
    
    sizeIn = env.observation_space.shape[0]     #4  = len(phi(x)) #cart_pos, cart_spe, pole_pos, pole_spe
    sizeOut = env.action_space.n                #[0, 1] # gauche droite
    
    Q = utils.NN_Q(sizeIn, sizeOut, [200])         #type: nn.Module
    V = utils.NN_V(sizeIn, [200])       #type: nn.Module
    t_max = 5000
    agent = BatchA2C_Agent(t_max, sizeOut, sizeIn, Q, V,
                           writer=writer, verbose=V_BENCH + V_PLOT + V_GRAD)
    
    loop(env, agent, writer, episode_count=1001, render_sleep=0., verbose=3)
    
    env.close()


if __name__ == '__main__':
    main_cartPole()
    # main_acrobot()
    # main_mountainCar()
    # main_lunarlander()
