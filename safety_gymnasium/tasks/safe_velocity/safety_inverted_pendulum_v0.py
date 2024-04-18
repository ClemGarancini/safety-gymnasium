# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ant environment with a safety constraint on velocity."""


import numpy as np
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv


class SafetyInvertedPendulumEnv(InvertedPendulumEnv):
    """Inverted pendulum environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # TO Define
        # self._velocity_threshold = 2.5745
        self._velocity_cost_gain = 1.0
        self._angular_threshold = (
            5.0 * np.pi / 180.0
        )  # threshold of 5 degrees (theta in radiant)
        self._angular_velocity_cost_gain = 1.0

    def step(self, action):  # pylint: disable=too-many-locals
        reward = 1.0
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))

        velocity_cost = self._velocity_cost_gain * ob[2]
        angular_cost = float(np.abs(ob[1]) > self._angular_threshold)
        angular_velocity_cost = self._angular_velocity_cost_gain * ob[3]
        cost = velocity_cost + angular_cost + angular_velocity_cost

        info = {
            "reward_survive": reward,
            "velocity_cost": velocity_cost,
            "angular_cost": angular_cost,
            "angular_velocity_cost": angular_velocity_cost,
            "cost": cost,
        }

        if self.render_mode == "human":
            self.render()
        return ob, reward, cost, terminated, False, info
