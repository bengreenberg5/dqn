from datetime import datetime
import gin
import os
import numpy as np
from tqdm import tqdm
import wandb

import gym
from gym.wrappers import AtariPreprocessing
import torch
import torch.nn.functional as F

from agent import DQNAgent
from replay import ReplayBuffer, Experience


ATARI_ENVS = ["Breakout"]


def preprocess_env(env):
    return AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )


def batchify(state, add_channel_dim=False):
    if not add_channel_dim:
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    else:
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def evaluate(env_name, agent, is_atari, epsilon, path=None):
    """
    :param env:
    :param agent:
    :param is_atari:
    :param epsilon:
    :param path: Path to video (if None, disable recording)
    :return:
    """
    env = gym.make(env_name)
    if is_atari:
        env = preprocess_env(env)
    if path is not None:
        recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, enabled=True, path=path)
    state = batchify(env.reset(), is_atari)
    rewards = 0.0
    done = False
    while not done:
        if path is not None:
            recorder.capture_frame()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_best_action(state, target=False).item()
        state, reward, done, _ = env.step(action)
        state = batchify(state, is_atari)
        rewards += reward
    if path is not None:
        recorder.close()
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
        env = preprocess_env(env)
    else:
        network_type = "linear"
    num_inputs = len(env.reset())
    agent = DQNAgent(
        num_inputs=num_inputs, num_outputs=env.action_space.n, network_type=network_type, device=device
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
            state_next = batchify(state_next, is_atari)
            replay_buffer.append(Experience(state, action, reward, state_next, done))
            ep_reward += reward
            state = state_next
            frame += 1
            progress_bar.update(1)

            # evaluate Q-net
            if frame >= eval_at:
                checkpoint = str(eval_at).zfill(7)
                checkpoint_dir = f"{dirname}/{checkpoint}/"
                os.mkdir(checkpoint_dir)
                rewards = [evaluate(
                    env_name, agent, is_atari=is_atari, epsilon=eval_epsilon, path=f"{checkpoint_dir}/{env_name}_{i}.mp4"
                ) for i in range(eval_episodes)]
                print(f"step {eval_at}: mean reward = {sum(rewards) / len(rewards):.2f}")
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
                rewards = torch.tensor([exp.reward for exp in exps]).to(device)
                state_nexts = torch.cat([exp.state_next for exp in exps]).float().to(device)
                dones = torch.tensor([exp.done for exp in exps]).to(device)

                agent.zero_grad()
                q_act_est = agent.est_values(states, actions, target=False)

                with torch.no_grad():
                    best_actions = (
                        agent.get_best_action(state_nexts, target=True)
                    )
                    q_eval_est = dones.logical_not() * agent.est_values(state_nexts, best_actions, target=True)

                loss = F.mse_loss(q_act_est, rewards + discount_factor * q_eval_est)
                loss.backward()
                agent.apply_grad()

        ep_rewards.append(ep_reward)
        if wandb.run:
            wandb.log({"reward": np.mean(ep_rewards[-25:])})

    progress_bar.close()
    return agent, ep_rewards


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v
    return data


def main():
    gin.parse_config_file("config.gin")
    config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)

    time = datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists("../runs"):
        os.mkdir("../runs")
    dirname = os.path.abspath(f"../runs/{time}/")
    os.mkdir(dirname)

    wandb.login()
    wandb.init(project="dqn", entity="anchorwatt", config=config_dict, dir=os.path.abspath(".."))

    train(dirname=dirname)

    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
