import numpy as np
import pytest

from smart_grid_rl.env import VirtualSmartHomeEnv


def test_reset_returns_correct_shape():
    env = VirtualSmartHomeEnv(seed=1)
    state, info = env.reset()
    assert state.shape == env.observation_space.shape
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)


def test_episode_terminates_after_steps_per_day():
    env = VirtualSmartHomeEnv(steps_per_day=10, seed=1)
    env.reset()
    done = False
    count = 0

    while not done:
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        count += 1

    assert count == 10


def test_step_rejects_wrong_action_length():
    env = VirtualSmartHomeEnv(seed=1)
    env.reset()

    with pytest.raises(ValueError):
        env.step(np.array([1]))  # Invalid action length