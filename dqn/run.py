import argparse
from datetime import datetime
import gin
import gym
import numpy as np
import os
from pprint import pprint
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from agent import DQNAgent
from replay import ReplayBuffer, Experience
from utils import *


def evaluate(env_name, agent, is_atari, epsilon, episodes=5, video_dir=None):
    """
    Evaluate an agent in a gym environment.
    :param env_name: name of environment
    :param agent: agent with two q-nets
    :param is_atari: whether or not this env requires atari preprocessing
    :param epiodes: how many trials to evaluate
    :param video_dir: where to store videos (default: don't record)
    :return: mean of episode rewards
    """
    env = gym.make(env_name)
    if is_atari:
        env = preprocess_env(env)
    if video_dir is not None:
        env = gym.wrappers.RecordVideo(
            env,
            episode_trigger=lambda _: True,
            video_folder=f"{video_dir}/videos",
            name_prefix=env_name.split("/")[-1],
        )
    rewards = 0.0
    for ep in range(episodes):
        state = batchify(env.reset(), is_atari)
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_best_action(state, target=False).item()
            state, reward, done, _ = env.step(action)
            state = batchify(state, is_atari)
            rewards += reward
    if video_dir is not None:
        env.close()
    return rewards / episodes


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
    Create and train a DQN agent.
    :param env_name: name of gym environment
    :param total_frames: how many frames to train for
    :param minibatch_size: how many experiences to sample
    :param exp_buffer_size:  how many experiences to store
    :param epsilon_start: exploration rate at beginning
    :param epsilon_end: exploration rate at end
    :param epsilon_decay_frames: how many frames to decay epsilon over
    :param replay_start_frame: how many frames to take random actions
    :param discount_factor: "gamma"
    :param train_every: how many frames in between sampling experiences
    :param update_target_every: how many gradient updates in between target network reset
    :param eval_every: how many frames in bewteen evaluations
    :param eval_epsilon: exploration rate for evaluation
    :param eval_episodes: how many episodes to evaluate
    :param dirname: where to store agent checkpoint + eval videos
    :param device: "cpu" or "cuda"
    :return: the trained agent and an array of training rewards
    """
    assert torch.cuda.is_available() or device == "cpu"
    env = gym.make(env_name)
    is_atari = np.any([env_name.startswith(atari_env) for atari_env in ATARI_ENVS])
    if is_atari:
        network_type = "conv"
        env = preprocess_env(env)
    else:
        network_type = "linear"
    num_inputs = len(env.reset())
    agent = DQNAgent(
        num_inputs=num_inputs,
        num_outputs=env.action_space.n,
        network_type=network_type,
        device=device,
    )
    replay_buffer = ReplayBuffer(exp_buffer_size)

    frame = 0
    progress_bar = tqdm(total=total_frames)
    ep_rewards = []
    eval_at = 0
    update_target_at = 0
    print(f"{datetime.now().strftime('%H:%M:%S')} - started training on {device}")

    while frame < total_frames:

        # set up episode
        state = batchify(env.reset(), is_atari)
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
                action = agent.get_best_action(state, target=False).item()
            state_next, reward, done, _ = env.step(action)
            state_next = batchify(state_next, add_channel_dim=is_atari)
            replay_buffer.append(
                Experience(state, action, reward, state_next, done), cast_to_int=True
            )
            ep_reward += reward
            state = state_next
            frame += 1
            progress_bar.update(1)

            # evaluate Q-net
            if frame >= eval_at:
                checkpoint = str(eval_at).zfill(8)
                checkpoint_dir = f"{dirname}/{checkpoint}"
                os.mkdir(checkpoint_dir)
                mean_rewards = evaluate(
                    env_name,
                    agent,
                    is_atari,
                    eval_epsilon,
                    episodes=eval_episodes,
                    video_dir=checkpoint_dir,
                )
                print(
                    f"step {eval_at}: episode {len(ep_rewards)}, mean reward = {mean_rewards:.2f}"
                )
                eval_at += eval_every
                agent.save(dirname, checkpoint)

            # update evaluation Q-net
            if frame > update_target_at:
                agent.update_target()
                update_target_at += train_every * update_target_every

            # train Q-net
            if frame % train_every == 0 and len(replay_buffer) >= minibatch_size:
                exps = replay_buffer.sample_experience(minibatch_size)
                states = torch.cat([exp.state for exp in exps]).float().to(device)
                actions = torch.tensor([exp.action for exp in exps]).to(device)
                rewards = torch.tensor([exp.reward for exp in exps]).float().to(device)
                state_nexts = (
                    torch.cat([exp.state_next for exp in exps]).float().to(device)
                )
                dones = torch.tensor([exp.done for exp in exps]).to(device)

                agent.zero_grad()
                q_act_est = agent.est_values(states, actions, target=False)

                with torch.no_grad():
                    best_actions = agent.get_best_action(state_nexts, target=True)
                    q_eval_est = dones.logical_not() * agent.est_values(
                        state_nexts, best_actions, target=True
                    )
                    q_eval_est = rewards + discount_factor * q_eval_est

                loss = F.mse_loss(q_act_est, q_eval_est)
                loss.backward()
                agent.apply_grad()

        ep_rewards.append(ep_reward)
        if wandb.run:
            recent_ep_rewards = ep_rewards[-100:]
            wandb.log(
                {
                    "training episode reward": sum(recent_ep_rewards)
                    / len(recent_ep_rewards)
                }
            )
            wandb.log({"epsilon": epsilon})

    progress_bar.close()
    return agent, ep_rewards


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parse the gin configuration to a dictionary.
    :param gin_config: the gin's config dictionary; gin.config._OPERATIVE_CONFIG
    :return: the parsed dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="name of .gin file with (hyper)parameters")
    parser.add_argument("--env", help="name of environment")
    args = parser.parse_args()
    gin.parse_config_file(os.path.abspath(f"../configs/{args.config}.gin"))
    config_dict = gin_config_to_readable_dictionary(gin.config._CONFIG)
    pprint(config_dict)

    time = datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists("../runs"):
        os.mkdir("../runs")
    dirname = os.path.abspath(f"../runs/{time}/")
    os.mkdir(dirname)

    wandb.login()
    wandb.init(
        project="dqn",
        entity="anchorwatt",
        config=config_dict,
        monitor_gym=True,
        dir=os.path.abspath(".."),
    )

    train(env_name=args.env, dirname=dirname)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
