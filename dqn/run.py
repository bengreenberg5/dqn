import argparse

import gym
import numpy as np
import random
import torch

from agent import DQNAgent, RandomAgent
from replay import ReplayBuffer, Experience


def preprocess_images(images, env_name):
    images = np.stack(images)
    if env_name.startswith("Breakout"):
        pass  # TODO
    else:
        return torch.tensor(images)


def process_action(env, env_name, action, history_length):
    images = []
    reward = 0
    done = False
    # aggregate images into state
    for _ in range(history_length):
        image_next, r, done, _ = env.step(action)
        images.append(image_next)
        reward += r
    state_next = preprocess_images(images, env_name)
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
):
    agent = DQNAgent(env_name, learning_rate, momentum, discount_factor)
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
            epsilon = epsilon_init * (1 - frame / epsilon_final_frame) + epsilon_final * (frame / epsilon_final_frame)

        # setup episode
        state = first_state(env, env_name, history_length)
        ep_reward = 0
        done = False
        while not done:

            if ep % 100 == 0:
                env.render()

            # take one action (for at least one frame)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            state_next, reward, done = process_action(env, env_name, action, history_length)
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
        if ep % 100 == 0 and ep > 0:
            last_100 = sum(ep_rewards[-100:]) / 100
            total = sum(ep_rewards) / len(ep_rewards)
            print(f"episode {ep}, " f"mean reward (last 100) = {last_100}, " f"mean reward (total) = {total:.3f}, frame {frame}")

    return agent


def main():
    # training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", help="gym environment name")
    parser.add_argument("--history_length", default=1, help="number of recent frames input to Q-networks")
    parser.add_argument("--num_episodes", default=10_000, help="how many episodes to train for")
    parser.add_argument("--minibatch_size", default=32, help="number of experiences per gradient step")
    parser.add_argument("--exp_buffer_size", default=1_000_000, help="number of experiences to store")
    parser.add_argument("--epsilon_init", default=1, help="initial exploration value")
    parser.add_argument("--epsilon_final", default=0.1, help="final exploration value")
    parser.add_argument("--epsilon_final_frame", default=1_000_000, help="the number of frames over which epsilon is linearly annealed to its final value")
    parser.add_argument("--replay_start_frame", default=50_000, help="how many frames of random play before learning starts")
    parser.add_argument("--q_target_update_freq", default=1, help="initial exploration coefficient")
    parser.add_argument("--learning_rate", default=2.5e-4, help="optimizer learning rate")
    parser.add_argument("--momentum", default=0.95, help="alpha / optimizer learning rate")
    parser.add_argument("--discount_factor", default=0.99, help="gamma / target discount factor")
    args = parser.parse_args()

    env_name = args.env
    env = gym.make(env_name)

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
    )

    # run episodes
    rewards = []
    frames = []
    for ep in range(100):
        state = first_state(env, env_name, args.history_length)
        ep_reward = 0
        done = False
        frame = 0
        while not done:
            if random.random() < args.epsilon_final:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            state_next, reward, done = process_action(env, env_name, action, args.history_length)
            ep_reward += reward
            if ep % 5 == 0:
                env.render()
            state = state_next
            frame += args.history_length
        rewards.append(ep_reward)
        frames.append(frame)
        print(ep, ep_reward, frame)

    print(sum(rewards) / len(rewards))
    print(sum(frames) / len(frames))


if __name__ == "__main__":
    main()
