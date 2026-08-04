"""Physical and economic component models for the virtual smart home."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class ApplianceState(Enum):
    OFF = 0
    ON = 1
    DEFERRED = 2


@dataclass
class Appliance:
    """A single controllable household appliance."""

    name: str
    rated_power_kw: float
    deferrable: bool = False
    min_run_steps: int = 1
    state: ApplianceState = ApplianceState.OFF
    steps_running: int = 0

    def __post_init__(self) -> None:
        if self.rated_power_kw <= 0:
            raise ValueError(f"{self.name}: rated_power_kw must be positive")
        if self.min_run_steps < 1:
            raise ValueError(f"{self.name}: min_run_steps must be >= 1")

    def turn_on(self) -> None:
        self.state = ApplianceState.ON
        self.steps_running = 0

    def turn_off(self) -> None:
        # Prevent short-cycling: block OFF before the minimum run duration
        # elapses if the appliance is currently ON.
        if self.state == ApplianceState.ON and self.steps_running < self.min_run_steps:
            return
        self.state = ApplianceState.OFF
        self.steps_running = 0

    def defer(self) -> None:
        if not self.deferrable:
            raise ValueError(f"{self.name} is not deferrable")
        self.state = ApplianceState.DEFERRED

    def tick(self) -> float:
        """Advance one time step; returns power drawn this step (kW)."""
        if self.state == ApplianceState.ON:
            self.steps_running += 1
            return self.rated_power_kw
        return 0.0


@dataclass
class BatteryStorage:
    """Battery Energy Storage System (BESS) with SoC and degradation tracking."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    soc_kwh: float = field(default=0.0)
    round_trip_efficiency: float = 0.95
    min_soc_fraction: float = 0.10
    max_soc_fraction: float = 0.95
    cumulative_throughput_kwh: float = 0.0  # proxy for degradation

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not (0.0 <= self.round_trip_efficiency <= 1.0):
            raise ValueError("round_trip_efficiency must be within [0, 1]")
        if self.soc_kwh == 0.0:
            self.soc_kwh = 0.5 * self.capacity_kwh

    @property
    def soc_fraction(self) -> float:
        return self.soc_kwh / self.capacity_kwh

    @property
    def min_soc_kwh(self) -> float:
        return self.min_soc_fraction * self.capacity_kwh

    @property
    def max_soc_kwh(self) -> float:
        return self.max_soc_fraction * self.capacity_kwh

    def charge(self, power_kw: float, dt_hours: float) -> float:
        """Charge the battery; returns actual power accepted (kW)."""
        if power_kw < 0:
            raise ValueError("power_kw must be non-negative for charge()")
        power_kw = min(power_kw, self.max_charge_kw)
        headroom_kwh = self.max_soc_kwh - self.soc_kwh
        max_energy_in = headroom_kwh / max(self.round_trip_efficiency, 1e-6)
        energy_requested = power_kw * dt_hours
        energy_accepted = min(energy_requested, max_energy_in)
        self.soc_kwh += energy_accepted * self.round_trip_efficiency
        self.cumulative_throughput_kwh += energy_accepted
        return energy_accepted / dt_hours if dt_hours > 0 else 0.0

    def discharge(self, power_kw: float, dt_hours: float) -> float:
        """Discharge the battery; returns actual power delivered (kW)."""
        if power_kw < 0:
            raise ValueError("power_kw must be non-negative for discharge()")
        power_kw = min(power_kw, self.max_discharge_kw)
        available_kwh = self.soc_kwh - self.min_soc_kwh
        energy_requested = power_kw * dt_hours
        energy_delivered = min(energy_requested, max(available_kwh, 0.0))
        self.soc_kwh -= energy_delivered
        self.cumulative_throughput_kwh += energy_delivered
        return energy_delivered / dt_hours if dt_hours > 0 else 0.0

    def degradation_index(self) -> float:
        """Simple proxy: full-equivalent-cycles completed (higher = more wear)."""
        return self.cumulative_throughput_kwh / (2 * self.capacity_kwh)


@dataclass
class RenewableSource:
    """Synthetic solar + wind generation profile."""

    solar_capacity_kw: float
    wind_capacity_kw: float
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = None
        self._init_rng()

    def _init_rng(self) -> None:
        import numpy as np

        self._rng = np.random.default_rng(self.seed)

    def generation_kw(self, step_of_day: int, steps_per_day: int) -> float:
        """Return combined renewable generation (kW) for a given time step."""
        hour = (step_of_day / steps_per_day) * 24.0
        # Solar: bell curve centered at noon, zero at night.
        solar = max(0.0, math.sin(math.pi * (hour - 6) / 12)) * self.solar_capacity_kw
        solar *= 1.0 + 0.05 * self._rng.standard_normal()
        # Wind: noisy baseline, largely independent of time of day.
        wind = (0.4 + 0.15 * self._rng.standard_normal()) * self.wind_capacity_kw
        return max(0.0, solar) + max(0.0, wind)


@dataclass
class GridModel:
    """Time-of-use pricing and overload threshold."""

    base_price_per_kwh: float = 0.15
    peak_price_per_kwh: float = 0.35
    peak_hours: tuple[int, int] = (17, 21)  # 5 PM - 9 PM
    overload_threshold_kw: float = 8.0

    def price(self, step_of_day: int, steps_per_day: int) -> float:
        hour = (step_of_day / steps_per_day) * 24.0
        if self.peak_hours[0] <= hour < self.peak_hours[1]:
            return self.peak_price_per_kwh
        return self.base_price_per_kwh

    def is_overloaded(self, total_demand_kw: float) -> bool:
        return total_demand_kw > self.overload_threshold_kw