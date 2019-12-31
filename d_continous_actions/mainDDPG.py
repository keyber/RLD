import numpy as np
import torch
import gym
import argparse
import os

from DDPG import Actor, Critic, DDPG, ReplayBuffer

import matplotlib.pyplot as plt


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.act(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCarContinuous-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--render_interval", default=10)
    parser.add_argument("--max_episodes", default=500)
    parser.add_argument("--start_episode", default=50)
    args = parser.parse_args()

    file_name = f"{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    kwargs = {
        "env": env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "min_action": min_action,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    policy = DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")


    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    # state, done = env.reset(), False
    # episode_reward = 0
    # episode_timesteps = 0
    # episode_num = 0


    lreward = []

    # train DDPG
    for i_episode in range(int(args.max_episodes)):
        state, done = env.reset(), False
        policy.reset_noise()
        render = i_episode % args.render_interval == 0

        episode_reward = 0
        episode_timesteps = 0

        while True:
            episode_timesteps += 1

            # Select action randomly or according to policy
            warmup = i_episode < args.start_episode
            action = policy.act(state, warmup)
            env.render()

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 1
            policy.update(next_state, reward, done_bool, args.batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Episode Num: {i_episode + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        lreward.append(episode_reward)

        # Evaluate episode
        if (i_episode + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")


    # plt.figure(figsize=(10, 8))
    # plt.subplot(2, 1, 1)
    plt.scatter(np.arange(args.max_episodes), lreward)
    plt.show()

    # plt.subplot(2, 1, 2)
    plt.plot(evaluations)

    plt.show()
