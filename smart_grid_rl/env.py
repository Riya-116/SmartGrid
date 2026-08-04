"""Gymnasium-compatible virtual smart home environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smart_grid_rl.components import Appliance, BatteryStorage, GridModel, RenewableSource
from smart_grid_rl.state_encoder import encode_state, state_vector_length

# Action encoding per appliance: 0 = OFF, 1 = ON, 2 = DEFER (if deferrable)
# Battery action is appended last: 0 = idle, 1 = charge, 2 = discharge
NUM_APPLIANCE_ACTIONS = 3
NUM_BATTERY_ACTIONS = 3


class VirtualSmartHomeEnv(gym.Env):
    """Simulates a smart home with appliances, battery storage, renewables, and grid pricing."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        steps_per_day: int = 96,
        dt_hours: float = 0.25,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

        self.steps_per_day = steps_per_day
        self.dt_hours = dt_hours
        self._episode_seed = seed

        self.appliances: list[Appliance] = [
            Appliance(name="HVAC", rated_power_kw=2.5, deferrable=False, min_run_steps=2),
            Appliance(name="Lights", rated_power_kw=0.3, deferrable=False),
            Appliance(name="Washer", rated_power_kw=1.2, deferrable=True, min_run_steps=3),
            Appliance(name="EV", rated_power_kw=3.5, deferrable=True, min_run_steps=4),
        ]
        self.battery = BatteryStorage(
            capacity_kwh=13.5, max_charge_kw=5.0, max_discharge_kw=5.0
        )
        self.renewable = RenewableSource(
            solar_capacity_kw=6.0, wind_capacity_kw=2.0, seed=seed or 42
        )
        self.grid = GridModel()

        self.current_step = 0

        n_appliances = len(self.appliances)
        self.action_space = spaces.MultiDiscrete(
            [NUM_APPLIANCE_ACTIONS] * n_appliances + [NUM_BATTERY_ACTIONS]
        )

        obs_len = state_vector_length(n_appliances)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        for appliance in self.appliances:
            appliance.turn_off()
        self.battery.soc_kwh = 0.5 * self.battery.capacity_kwh
        self.battery.cumulative_throughput_kwh = 0.0
        renewable_kw = self.renewable.generation_kw(self.current_step, self.steps_per_day)
        state = encode_state(
            self.appliances, self.battery, renewable_kw, self.grid,
            self.current_step, self.steps_per_day,
        )
        return state, {"renewable_kw": renewable_kw}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if len(action) != len(self.appliances) + 1:
            raise ValueError(
                f"Expected action of length {len(self.appliances) + 1}, got {len(action)}"
            )

        # Apply appliance actions.
        for appliance, act in zip(self.appliances, action[:-1]):
            if act == 0:
                appliance.turn_off()
            elif act == 1:
                appliance.turn_on()
            elif act == 2 and appliance.deferrable:
                appliance.defer()
            # Invalid combos (e.g. DEFER on non-deferrable) silently no-op;
            # the agent is expected to learn a valid policy via reward shaping (Day 3).

        appliance_load_kw = sum(a.tick() for a in self.appliances)

        renewable_kw = self.renewable.generation_kw(self.current_step, self.steps_per_day)

        # Apply battery action.
        battery_act = int(action[-1])
        battery_power_kw = 0.0
        if battery_act == 1:  # charge
            surplus = max(renewable_kw - appliance_load_kw, self.battery.max_charge_kw)
            battery_power_kw = self.battery.charge(surplus, self.dt_hours)
        elif battery_act == 2:  # discharge
            deficit = max(appliance_load_kw - renewable_kw, self.battery.max_discharge_kw)
            battery_power_kw = self.battery.discharge(deficit, self.dt_hours)

        net_grid_load_kw = appliance_load_kw - renewable_kw + (
            battery_power_kw if battery_act == 1 else -battery_power_kw if battery_act == 2 else 0.0
        )
        net_grid_load_kw = max(net_grid_load_kw, 0.0)
        overloaded = self.grid.is_overloaded(net_grid_load_kw)

        self.current_step += 1
        done = self.current_step >= self.steps_per_day

        state = encode_state(
            self.appliances, self.battery, renewable_kw, self.grid,
            self.current_step, self.steps_per_day,
        )

        # Reward is finalized on Day 3; stubbed to 0.0 for now so the
        # environment loop is testable end-to-end today.
        reward = 0.0

        info = {
            "appliance_load_kw": appliance_load_kw,
            "renewable_kw": renewable_kw,
            "net_grid_load_kw": net_grid_load_kw,
            "overloaded": overloaded,
            "battery_soc_fraction": self.battery.soc_fraction,
            "price_per_kwh": self.grid.price(self.current_step, self.steps_per_day),
        }
        return state, reward, done, False, info