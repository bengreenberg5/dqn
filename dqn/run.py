import argparse
import cv2
from datetime import datetime
import os
import numpy as np
import random

import gym
from gym.wrappers import Monitor
import torch

# from agent import DQNAgent
# from replay import ReplayBuffer, Experience


def preprocess_images(images, env_name, prev_image=None):

    def rescale(image):
        r = cv2.resize(image[0], (84, 84))
        g = cv2.resize(image[1], (84, 84))
        b = cv2.resize(image[2], (84, 84))
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return np.stack([r, g, b, y], axis=0)

    images = [torch.tensor(rescale(image)).float() for image in images]
    processed_images = []
    if env_name in ["Breakout-v4"]:
        # N, color channels, frame depth, frame height, frame width
        # 1,              4,           4,           84,          84
        if prev_image is not None:
            first_image = torch.maximum(prev_image[0, :, -1, :, :], images[0])
        else:
            first_image = images[0]
        processed_images.append(first_image)
        for i in range(1, len(images)):
            processed_images.append(torch.maximum(images[i-1], images[i]))
        processed_images = torch.stack(processed_images, dim=1).unsqueeze(0)

    else:
        processed_images = torch.tensor(images)

    return processed_images


def process_action(env, env_name, action, history_length, prev_image=None):
    images = []
    reward = 0
    done = False
    # aggregate images into state
    for _ in range(history_length):
        if not done:
            image_next, r, done, _ = env.step(action)
        images.append(image_next)
        reward += r
    state_next = preprocess_images(images, env_name, prev_image)
    return state_next, reward, done


def first_state(env, env_name, history_length):
    # duplicate state `history_length` times... state is history_length * H * W
    image = env.reset()
    images = [image] * history_length
    state = preprocess_images(images, env_name)
    return state


def train(
    env,
    env_name,
    history_length,
    num_episodes,
    minibatch_size=32,
    exp_buffer_size=100_000,
    epsilon_init=1.0,
    epsilon_final=0.1,
    epsilon_final_frame=100_000,
    replay_start_frame=5_000,
    q_target_update_freq=1_000,
    learning_rate=1e-4,
    momentum=0.0,
    discount_factor=0.99,
    save_every=0,
    dirname="",
):
    agent = DQNAgent(env_name, history_length, learning_rate, momentum, discount_factor)
    replay = ReplayBuffer(exp_buffer_size)
    frame = 0
    ep_rewards = []

    for ep in range(num_episodes):

        # get probability of random action
        if frame < replay_start_frame:
            epsilon = epsilon_init
        elif frame >= epsilon_final_frame:
            epsilon = epsilon_final
        else:
            epsilon = epsilon_init * (
                1 - frame / epsilon_final_frame
            ) + epsilon_final * (frame / epsilon_final_frame)

        # setup episode
        state = first_state(env, env_name, history_length)
        ep_reward = 0
        done = False

        while not done:

            # take one action (for at least one frame)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            state_next, reward, done = process_action(
                env, env_name, action, history_length, prev_image=state
            )
            ep_reward += reward
            frame += history_length

            # update gradients using experience replay
            replay.append(Experience(state, action, reward, state_next, done))
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
                if (frame / history_length) % q_target_update_freq == 0:
                    agent.reset_target()

            # update state
            state = state_next

        ep_rewards.append(ep_reward)

        if save_every > 0 and ep % save_every == 0:
            agent.save_networks(f"{dirname}/{str(ep).zfill(7)}")

        if save_every > 0 and ep > 0 and ep % save_every == 0:
            last = sum(ep_rewards[-save_every:]) / save_every
            total = sum(ep_rewards) / len(ep_rewards)
            print(
                f"episode {ep}, "
                f"mean reward (last {save_every}) = {last}, "
                f"mean reward (total) = {total:.3f}, frame {frame}"
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

    agent = train(
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

    # # run episodes
    # rewards = []
    # frames = []
    # for _ in range(100):
    #     state = first_state(env, env_name, args.history_length)
    #     ep_reward = 0
    #     done = False
    #     frame = 0
    #     while not done:
    #         if random.random() < args.epsilon_final:
    #             action = env.action_space.sample()
    #         else:
    #             action = agent.get_action(state)
    #         state_next, reward, done = process_action(
    #             env, env_name, action, args.history_length, prev_image=state
    #         )
    #         ep_reward += reward
    #         state = state_next
    #         frame += args.history_length
    #     rewards.append(ep_reward)
    #     frames.append(frame)
    #     print(ep_reward, frame)
    #
    # print(sum(rewards) / len(rewards))
    # print(sum(frames) / len(frames))


if __name__ == "__main__":
    main()
