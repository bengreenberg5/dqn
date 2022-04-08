import argparse
import gym
import numpy as np

from dqn.agent import *
from dqn.utils import ATARI_ENVS, batchify, preprocess_env


def evaluate(env_name, env, run_dir, checkpoint, epsilon, episodes=5):
    """
    Evaluate agent from checkpoint; print rewards and save videos.
    """
    assert os.path.exists(run_dir), f"can't find {run_dir}"
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda _: True,
        video_folder=f"{run_dir}/{checkpoint}/videos",
        name_prefix=env_name,
    )
    ep_rewards = []
    for ep in range(episodes):
        rewards = 0.0
        state = batchify(env.reset(), add_channel_dim=True)
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_best_action(state, target=False).item()
            state, reward, done, _ = env.step(action)
            state = batchify(state, add_channel_dim=True)
            rewards += reward
        ep_rewards.append(rewards)
    env.close()
    return ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="directory containing checkpoint files")
    parser.add_argument("--env", help="name of gym env")
    args = parser.parse_args()
    run_dir = f"runs/{args.run}"

    env_name = args.env
    is_atari = np.any([env_name.startswith(atari_env) for atari_env in ATARI_ENVS])
    network_type = "conv" if is_atari else "linear"
    env = preprocess_env(gym.make(env_name), episodic_life=False)
    agent = DQNAgent(
        network_type=network_type,
        num_inputs=4,
        num_outputs=env.action_space.n,
        device="cpu",
    )
    checkpoints = sorted(os.listdir(run_dir))
    out_file = open(f"{run_dir}/{env_name}_video_rewards.txt", "a")
    checkpoints = [dir for dir in os.listdir(run_dir) if dir[0].isdigit()]
    for checkpoint in checkpoints:
        agent.load(run_dir, checkpoint)
        print(f"loaded checkpoint {checkpoint} from {run_dir}")
        ep_rewards = evaluate(
            env_name,
            env,
            run_dir,
            checkpoint,
            epsilon=0.05,
            episodes=10,
        )
        reward_str = f"{run_dir}/{checkpoint}: {ep_rewards}\n"
        out_file.write(reward_str)
    out_file.close()
