import argparse
import gym
import numpy as np

from dqn.agent import *
from dqn.utils import batchify, preprocess_env


def evaluate(env, run_dir, checkpoint, epsilon, episodes=5):
    assert os.path.exists(run_dir), f"can't find {run_dir}"
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda _: True,
        video_folder=f"{run_dir}/{checkpoint}/videos",
        name_prefix="Breakout",
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
    args = parser.parse_args()
    run_dir = f"runs/{args.run}"

    env_name = "BreakoutDeterministic-v4"
    env = preprocess_env(gym.make(env_name), episodic_life=False)
    agent = DQNAgent(
        network_type="conv",
        num_inputs=4,
        num_outputs=4,
        device="cpu",
    )
    checkpoints = sorted(os.listdir(run_dir))
    out = ""
    out_file = open("video_rewards.txt", "w")
    for checkpoint in os.listdir(run_dir):
        agent.load(run_dir, checkpoint)
        ep_rewards = evaluate(
            env,
            run_dir,
            checkpoint,
            epsilon=0.05,
            episodes=5,
        )
        reward_str = f"{run_dir}/{checkpoint}: {ep_rewards}\n"
        print(reward_str)
        out += reward_str
    out_file.write(out)
    out_file.close()




