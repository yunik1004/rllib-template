"""Train script.
If you want to use custom policy algorithm, refer to https://docs.ray.io/en/latest/rllib/rllib-concepts.html
"""

import argparse
import os
import ray
from ray import air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import Tuner
from package_name.callbacks.test import CustomCallbacks
from package_name.environments.test import CustomEnv
from package_name.models.test import CustomModel


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    ray.init()

    config = (
        PPOConfig()
        .framework(framework="torch")
        .environment(env=CustomEnv)
        # .api_stack(
        #    enable_rl_module_and_learner=True,
        #    enable_env_runner_and_connector_v2=True,
        # )
        .callbacks(callbacks_class=CustomCallbacks)
        .training(
            model={
                # "uses_new_env_runners": True,
                "custom_model": CustomModel,
                "custom_model_config": {},
            },
            train_batch_size=args.batch_size,
            lr=args.lr,
        )
        .checkpointing(export_native_model_files=True)
    )

    run_config = air.RunConfig(
        storage_path=args.log_dir,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=args.checkpoint_freq,
        ),
        stop={"training_iteration": args.max_iters},
    )

    tuner = Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )
    tuner.fit()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Train script")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of GPU nodes")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the model"
    )
    parser.add_argument(
        "--resume_train", action="store_true", help="resume training state"
    )
    parser.add_argument("--log_dir", type=str, default="~/ray_results", help="log dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--max_iters", type=int, default=1, help="maximum number of iterations"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=0, help="checkpoint frequency"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for data loading"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    opts.log_dir = os.path.abspath(opts.log_dir)  # Convert rel path to abs path
    main(opts)
