import argparse
from collections import deque
from copy import deepcopy
import cv2
from datetime import datetime
import os
import numpy as np
import random
from tqdm import tqdm

import gym
from gym.wrappers import Monitor, AtariPreprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import DQNAgent
from replay import ReplayBuffer, Experience


def evaluate(
    env,
    agent,
    epsilon,
):
    """

    :param env:
    :param agent:
    :param epsilon:
    :return:
    """
    return 0.0  # TODO evaluate and save videos


def make_policy():
    """
    TODO
    :return:
    """
    pass


def train(
    env_name,
    total_frames,
    minibatch_size=32,
    exp_buffer_size=1_000_000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_frames=1_000_000,
    replay_start_frame=50_000,
    learning_rate=2.5e-4,
    momentum=0.95,
    discount_factor=0.99,
    train_every=4,
    update_target_every=10_000,
    save_every=50_000,
    eval_every=10_000,
    eval_epsilon=0.05,
    eval_episodes=10,
    dirname="",
    device="cpu",
):
    """
    TODO
    :param env_name:
    :param total_frames:
    :param minibatch_size:
    :param exp_buffer_size:
    :param epsilon_start:
    :param epsilon_end:
    :param epsilon_decay_frames:
    :param replay_start_frame:
    :param learning_rate:
    :param momentum:
    :param discount_factor:
    :param train_every:
    :param update_target_every:
    :param save_every:
    :param eval_every:
    :param eval_epsilon:
    :param eval_episodes:
    :param dirname:
    :param device:
    :return:
    """
    env = gym.make(env_name)
    agent = DQNAgent(...)  # TODO
    replay_buffer = ReplayBuffer(exp_buffer_size)

    if save_every > 0:
        env = Monitor(
            env,
            f"{dirname}/videos",
            video_callable=lambda ep: ep % save_every == 0,
            force=True,
        )
        # video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env)
    if env_name.startswith("Breakout"):
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
        )

    frame = 0
    pbar = tqdm(total=total_frames)
    ep_rewards = []
    eval_at = 0
    update_target_at = 0
    print(f"{datetime.now().strftime('%H:%M:%S')} - started training on {device}")

    while frame < total_frames:
        if frame >= eval_at:
            eval_env = env  # monitor_env if eval_at % render_every == 0 else env
            rewards = [
                evaluate(eval_env, agent, epsilon=eval_epsilon)
                for _ in range(eval_episodes)
            ]  # TODO evaluate
            print(f"step {eval_at}: mean reward = {sum(rewards) / len(rewards):.2f}")
            eval_at += eval_every
        if frame > update_target_at:
            agent.reset_target()
            update_target_at += train_every * update_target_every
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        ep_reward = 0.0
        if frame >= replay_start_frame:
            epsilon_decay_frac = min(frame, epsilon_decay_frames) / epsilon_decay_frames
            epsilon = epsilon_start + epsilon_decay_frac * (epsilon_end - epsilon_start)
        else:
            epsilon = 1.0

        while not done:

            # run episode
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            state_next, reward, done, _ = env.step(action)
            state_next = torch.tensor(state_next)
            replay_buffer.append(Experience(state, action, reward, state_next, done))
            ep_reward += reward
            state = state_next
            frame += 1
            pbar.update(1)

            # train Q-net
            if frame % train_every == 0 and len(replay_buffer) >= minibatch_size:
                exps = replay_buffer.sample_experience(minibatch_size)
                states = torch.cat([exp.state for exp in exps]).to(device)
                actions = torch.tensor([exp.action for exp in exps]).to(device)
                rewards = torch.tensor([exp.reward for exp in exps]).to(device)
                state_nexts = torch.cat([exp.state_next for exp in exps]).to(device)
                dones = torch.tensor([exp.done for exp in exps]).to(device)

                q_value_est = agent.q_value_estimate(states).gather(1, actions.unsqueeze(-1)).squeeze()
                best_actions = (
                    agent.q_value_estimate(state_nexts).detach().argmax(dim=1)
                )
                q_target_est = dones * agent.q_target_estimate(state_nexts).detach().gather(
                    1, best_actions.unsqueeze(-1)
                ).squeeze()
                loss = F.mse_loss(q_value_est, rewards + discount_factor * q_target_est)
                loss.backward()

        ep_rewards += ep_reward

    pbar.close()
    return agent, ep_rewards


def main():
    # training args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="CartPole-v1", type=str, help="gym environment name"
    )
    parser.add_argument(
        "--history_length",
        default=4,
        type=int,
        help="number of recent frames input to Q-networks",
    )
    parser.add_argument(
        "--num_episodes",
        default=10_000,
        type=int,
        help="how many episodes to train for",
    )
    parser.add_argument(
        "--minibatch_size",
        default=32,
        type=int,
        help="number of experiences per gradient step",
    )
    parser.add_argument(
        "--exp_buffer_size",
        default=1_000_000,
        type=int,
        help="number of experiences to store",
    )
    parser.add_argument(
        "--epsilon_init", default=1, type=int, help="initial exploration value"
    )
    parser.add_argument(
        "--epsilon_final", default=0.1, type=float, help="final exploration value"
    )
    parser.add_argument(
        "--epsilon_final_frame",
        default=1_000_000,
        type=int,
        help="the number of frames over which epsilon is linearly annealed to its final value",
    )
    parser.add_argument(
        "--replay_start_frame",
        default=50_000,
        type=int,
        help="how many frames of random play before learning starts",
    )
    parser.add_argument(
        "--q_target_update_freq",
        default=10_000,
        type=int,
        help="initial exploration coefficient",
    )
    parser.add_argument(
        "--learning_rate", default=2.5e-4, type=float, help="optimizer learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.95, type=float, help="alpha / optimizer learning rate"
    )
    parser.add_argument(
        "--discount_factor",
        default=0.99,
        type=float,
        help="gamma / target discount factor",
    )
    parser.add_argument(
        "--save_every",
        default=0,
        type=int,
        help="how often to save model weights and video (0 to not save)",
    )
    args = parser.parse_args()

    env_name = args.env

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = os.path.abspath(
        f"../runs/{time}_{env_name}_{args.history_length}_{args.num_episodes}_{args.minibatch_size}_"
        f"{args.exp_buffer_size}_{args.epsilon_init}_{args.epsilon_final}_{args.epsilon_final_frame}_"
        f"{args.replay_start_frame}_{args.q_target_update_freq}_{args.learning_rate}_{args.momentum}_"
        f"{args.discount_factor}"
    )

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    train(...)  # TODO

if __name__ == "__main__":
    main()
