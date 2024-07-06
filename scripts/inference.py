"""Inference script
"""

import argparse
from gymnasium.wrappers.record_video import RecordVideo
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from package_name.environments.test import CustomEnv


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    ray.init()

    if args.interactive:
        env = CustomEnv(render_mode="human")
    else:
        env = CustomEnv(render_mode="rgb_array")
        env = RecordVideo(env, args.output_dir)

    agent = Algorithm.from_checkpoint(checkpoint=args.checkpoint)

    observation, info = env.reset(seed=args.seed)
    step = 0
    total_reward = 0
    while True:
        step += 1

        action = agent.compute_single_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        print(f"[Step {step}] Reward: {reward} / Total reward: {total_reward}")

        if terminated or truncated or (args.max_steps >= 0 and args.max_steps <= step):
            break
    env.close()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="~/ray_results", help="output video dir"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="enable interactive mode"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="maximum number of steps. disabled if < -1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
