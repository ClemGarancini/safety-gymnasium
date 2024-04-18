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
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer


class SafetyCartPoleEnv(CartPoleEnv):
    """Ant environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # TO Define
        # self._velocity_threshold = 2.5745
        self._velocity_cost_gain = 1.0
        self._angular_threshold = (
            5.0 * np.pi / 180.0
        )  # threshold of 5 degrees (theta in radiant)
        self._angular_velocity_cost_gain = 1.0
        self.dt = 1e-3

    def step(self, action):  # pylint: disable=too-many-locals
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        velocity_cost = self._velocity_cost_gain * x_dot
        angular_cost = float(np.abs(theta) > self._angular_threshold)
        angular_velocity_cost = self._angular_velocity_cost_gain * theta_dot
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
        return (
            np.array(self.state, dtype=np.float32),
            reward,
            cost,
            terminated,
            False,
            info,
        )
