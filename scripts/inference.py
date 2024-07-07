"""Inference script
"""

import argparse
from lightning.fabric import Fabric
from lightning.pytorch import seed_everything
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import ray
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import torch
from package_name.environments.test import CustomEnv


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    seed_everything(args.seed, workers=True)

    fabric = Fabric(accelerator=args.accelerator, devices="auto", strategy="auto")
    fabric.launch()

    ray.init()

    if args.interactive:
        env = CustomEnv(render_mode="human")
    else:
        env = CustomEnv(render_mode="rgb_array")
        env = RecordVideo(env, args.output_dir)

    rl_module = RLModule.from_checkpoint(checkpoint_dir_path=args.checkpoint)
    rl_module.eval()

    rl_module = fabric.setup(rl_module)

    obs, info = env.reset(seed=args.seed)
    step = 0
    total_reward = 0
    with torch.no_grad():
        while True:
            step += 1

            input_dict = {
                Columns.OBS: torch.from_numpy(obs).unsqueeze(0).to(rl_module.device)
            }
            if not args.explore:
                rl_module_out = rl_module.forward_inference(input_dict)
            else:
                rl_module_out = rl_module.forward_exploration(input_dict)

            logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
            action = np.random.choice(env.action_space.n, p=softmax(logits[0]))

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            print(f"[Step {step}] Reward: {reward} / Total reward: {total_reward}")

            if (
                terminated
                or truncated
                or (args.max_steps >= 0 and args.max_steps <= step)
            ):
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
        "--accelerator", type=str, default="auto", help="training accelerator"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the RL module"
    )
    parser.add_argument(
        "--output_dir", type=str, default="~/ray_results", help="output video dir"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="enable interactive mode"
    )
    parser.add_argument(
        "--explore", action="store_true", help="explore during inference"
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
