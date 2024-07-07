"""Train script
"""

import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from package_name.callbacks.test import CustomCallbacks
from package_name.environments.test import CustomEnv
from package_name.models.test import CustomRLModule


class CustomLearner(PPOTorchLearner):
    pass


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
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            #    num_cpus_per_env_runner=1,
            #    num_gpus_per_env_runner=1,
        )
        .learners(
            num_learners=args.num_learners,
            #    num_cpus_per_learner=1,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .checkpointing(export_native_model_files=True)
        .debugging(seed=args.seed)
        .callbacks(callbacks_class=CustomCallbacks)
        .rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class=CustomRLModule,
            ),
        )
        .training(
            model={
                "uses_new_env_runners": True,
            },
            train_batch_size=args.batch_size,
            sgd_minibatch_size=args.minibatch_size,
            lr=args.lr,
            learner_class=CustomLearner,
        )
    )

    tune.run(
        run_or_experiment=PPO,
        config=config,
        storage_path=args.log_dir,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=args.checkpoint_freq,
        ),
        stop={"training_iteration": args.max_iters},
        restore=args.checkpoint if args.checkpoint != "" else None,
    )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser("Train script")
    parser.add_argument(
        "--num_env_runners",
        type=int,
        default=0,
        help="number of env runners for sampling",
    )
    parser.add_argument(
        "--num_learners", type=int, default=0, help="number of learners for training"
    )
    parser.add_argument(
        "--num_gpus_per_learner", type=int, default=0, help="number of GPUs per learner"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint of the experiment runner"
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
        "--minibatch_size", type=int, default=512, help="minibatch size"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    opts.log_dir = os.path.abspath(opts.log_dir)  # Convert rel path to abs path
    main(opts)
