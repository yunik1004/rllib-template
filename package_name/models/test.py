"""Define custom models.
Refer to https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_modelv2.py
"""

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule


class CustomRLModule(PPOTorchRLModule):
    pass
