import argparse
from collections import deque
import cv2
from datetime import datetime
from line_profiler import LineProfiler
from memory_profiler import profile
import os
import numpy as np
import random

import gym
from gym.wrappers import Monitor
import torch

from agent import DQNAgent
from replay import ReplayBuffer, Experience


def rescale(image, env_name):
    if env_name.startswith("CartPole"):
        return image
    else:
        r = cv2.resize(image[:, :, 0], (84, 84))
        g = cv2.resize(image[:, :, 1], (84, 84))
        b = cv2.resize(image[:, :, 2], (84, 84))
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return np.expand_dims(y, axis=0)


def train(
    env,
    env_name,
    history_length,
    num_episodes,
    minibatch_size=32,
    exp_buffer_size=1_000_000,
    epsilon_init=1.0,
    epsilon_final=0.1,
    epsilon_final_frame=1_000_000,
    replay_start_frame=50_000,
    q_target_update_freq=10_000,
    learning_rate=2.5e-4,
    momentum=0.95,
    discount_factor=0.99,
    save_every=0,
    dirname="",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    replay = ReplayBuffer(exp_buffer_size)
    frames = 0
    actions = []
    ep_rewards = []
    agent = DQNAgent(env_name, history_length, learning_rate, momentum, discount_factor)

    print(f"{datetime.now().strftime('%H:%M:%S')} - started training on {device}")
    for ep in range(num_episodes):
        # get probability of random action
        if frames < replay_start_frame:
            epsilon = epsilon_init
        elif frames >= epsilon_final_frame:
            epsilon = epsilon_final
        else:
            epsilon = epsilon_init * (
                1 - frames / epsilon_final_frame
            ) + epsilon_final * (frames / epsilon_final_frame)

        # setup episode
        image = torch.Tensor(rescale(env.reset(), env_name)).type(torch.uint8).to(device)
        frame_buffer = deque([image], maxlen=history_length)
        ep_frames = 0
        ep_actions = {action: 0 for action in range(env.action_space.n)}
        ep_reward = 0
        done = False
        state = None
        while not done:
            if ep_frames < history_length:
                if env_name.startswith("CartPole"):
                    action = env.action_space.sample()
                else:
                    action = 0  # NOOP
            elif random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            ep_actions[action] += 1

            action_reward = 0
            for _ in range(history_length):
                if not done:
                    image, reward, done, _ = env.step(action)
                    image = torch.Tensor(rescale(image, env_name)).type(torch.uint8).to(device)
                    if ep_frames > 0:
                        image = torch.maximum(image, frame_buffer[-1])
                    action_reward += reward
                frame_buffer.append(image)
            ep_frames += history_length
            ep_reward += action_reward

            state_next = torch.stack(list(frame_buffer), dim=1).unsqueeze(0)

            # update gradients using experience replay
            if state is not None:
                replay.append(
                    Experience(state.to("cpu"), action, action_reward, state_next.to("cpu"), done)
                )

            if len(replay) > minibatch_size:
                exp_batch = replay.sample_experience(minibatch_size)
                target_estimate = agent.get_q_target_estimate(exp_batch)
                value_estimate = agent.get_q_value_estimate(exp_batch)
                loss = torch.nn.MSELoss()(value_estimate, target_estimate)

                # update gradient
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                # update target net
                if (frames / history_length) % q_target_update_freq == 0:
                    agent.reset_target()

            # update state
            state = state_next

        frames += ep_frames
        ep_rewards.append(ep_reward)
        actions.append(ep_actions)

        if ep % 10 == 0:
            print(
                f"{datetime.now().strftime('%H:%M:%S')} - frame {frames}, episode {ep}"
            )

        if save_every > 0 and ep % save_every == 0:
            agent.save_networks(f"{dirname}/{str(ep).zfill(7)}")
            if ep > 0:
                last = sum(ep_rewards[-save_every:]) / save_every
                total = sum(ep_rewards) / len(ep_rewards)
                print(
                    f"mean reward (last {save_every}) = {last}, "
                    f"mean reward (total) = {total:.3f}, "
                    f"last action distribution: {actions[-1]}"
                )

    return agent


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
    env = gym.make(env_name)

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = os.path.abspath(
        f"../runs/{time}_{env_name}_{args.history_length}_{args.num_episodes}_{args.minibatch_size}_"
        f"{args.exp_buffer_size}_{args.epsilon_init}_{args.epsilon_final}_{args.epsilon_final_frame}_"
        f"{args.replay_start_frame}_{args.q_target_update_freq}_{args.learning_rate}_{args.momentum}_"
        f"{args.discount_factor}"
    )

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if args.save_every > 0:
        env = Monitor(
            env,
            f"{dirname}/videos",
            video_callable=lambda ep: ep % args.save_every == 0,
            force=True,
        )

    lp = LineProfiler()
    lp_wrapper = lp(train)
    lp_wrapper(
        env=env,
        env_name=env_name,
        history_length=args.history_length,
        num_episodes=args.num_episodes,
        minibatch_size=args.minibatch_size,
        exp_buffer_size=args.exp_buffer_size,
        epsilon_init=args.epsilon_init,
        epsilon_final=args.epsilon_final,
        epsilon_final_frame=args.epsilon_final_frame,
        replay_start_frame=args.replay_start_frame,
        q_target_update_freq=args.q_target_update_freq,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        discount_factor=args.discount_factor,
        save_every=args.save_every,
        dirname=dirname,
    )
    lp.print_stats()


if __name__ == "__main__":
    main()
