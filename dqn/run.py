from datetime import datetime
import gin
import os
import numpy as np
from tqdm import tqdm

import gym
from gym.wrappers import Monitor, AtariPreprocessing
import torch
import torch.nn.functional as F

from agent import DQNAgent
from replay import ReplayBuffer, Experience


ATARI_ENVS = ["Breakout"]


def evaluate(env, agent, epsilon):
    """
    :param env:
    :param agent:
    :param epsilon:
    :return:
    """
    state = env.reset()
    rewards = 0.0
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.get_best_action(state, target=False)
        state, reward, done, _ = env.step(action)
        rewards += reward
    return rewards


@gin.configurable
def train(
    env_name,
    total_frames,
    minibatch_size=32,
    exp_buffer_size=1_000_000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_frames=1_000_000,
    replay_start_frame=50_000,
    discount_factor=0.99,
    train_every=4,
    update_target_every=10_000,
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
    :param discount_factor:
    :param train_every:
    :param update_target_every:
    :param eval_every:
    :param eval_epsilon:
    :param eval_episodes:
    :param dirname:
    :param device:
    :return:
    """
    env = gym.make(env_name)
    is_atari = np.any([env_name.startswith(atari_env) for atari_env in ATARI_ENVS])
    if is_atari:
        network_type = "conv"
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
        )
    else:
        network_type = "linear"
    agent = DQNAgent(network_type=network_type, device=device)
    replay_buffer = ReplayBuffer(exp_buffer_size)

    frame = 0
    pbar = tqdm(total=total_frames)
    ep_rewards = []
    eval_at = 0
    update_target_at = 0
    print(f"{datetime.now().strftime('%H:%M:%S')} - started training on {device}")

    while frame < total_frames:

        # set up episode
        if frame >= eval_at:
            checkpoint = str(eval_at).zfill(7)
            eval_env = Monitor(env, f"{dirname}/{checkpoint}/", force=True)
            # video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env)
            rewards = [evaluate(eval_env, agent, epsilon=eval_epsilon) for _ in range(eval_episodes)]
            print(f"step {eval_at}: mean reward = {sum(rewards) / len(rewards):.2f}")
            eval_at += eval_every
            agent.save(dirname, checkpoint)
        if frame > update_target_at:
            agent.update_target()
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
                action = agent.get_best_action(state, target=False)
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

                q_act_est = agent.est_values(states, actions, target=False)

                best_actions = (
                    agent.get_best_action(state_nexts, target=True).detach()
                )
                q_eval_est = dones * agent.est_values(state, best_actions, target=False)

                loss = F.mse_loss(q_act_est, rewards + discount_factor * q_eval_est)
                loss.backward()
                agent.apply_grad()

        ep_rewards += ep_reward

    pbar.close()
    return agent, ep_rewards


def main():
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = os.path.abspath(f"../runs/{time}/")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    train()

if __name__ == "__main__":
    main()
