"""Define environments.
Refer to https://docs.ray.io/en/latest/rllib/rllib-env.html
"""

from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CustomEnv(CartPoleEnv):
    pass
