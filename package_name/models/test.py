"""Define custom models.
Refer to https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_modelv2.py
"""

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class CustomModel(FullyConnectedNetwork):
    pass
