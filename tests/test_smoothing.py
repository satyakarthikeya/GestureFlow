"""Tests for core.smoothing and core.coordinate_mapper."""

import math
import time
import sys
from unittest.mock import MagicMock

# Mock third-party dependencies before importing core
for mod in [
    'cv2', 'pyautogui', 'screen_brightness_control',
    'pynput', 'pynput.keyboard', 'pycaw', 'pycaw.pycaw', 'comtypes', 'comtypes.stream',
    'mediapipe', 'mediapipe.python', 'mediapipe.python.solutions',
    'mediapipe.python.solutions.hands', 'mediapipe.python.solutions.drawing_utils',
    'mediapipe.python.solutions.drawing_styles'
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest

from core.smoothing import (
    AdaptiveCoordinateSmoother,
    CoordinateSmoother,
    ExponentialMovingAverage,
    OneEuroFilter,
    OneEuroFilter2D,
)
from core.coordinate_mapper import ScreenMapper


class TestEMA:
    def test_first_value(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        assert ema.update(10.0) == 10.0

    def test_convergence(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        for _ in range(100):
            result = ema.update(10.0)
        assert abs(result - 10.0) < 0.01

    def test_smoothing_effect(self):
        ema = ExponentialMovingAverage(alpha=0.3)
        ema.update(0.0)
        result = ema.update(100.0)
        # Should be much less than 100 due to smoothing
        assert result < 50.0

    def test_reset(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(100.0)
        ema.reset()
        assert ema.value is None
        assert ema.update(50.0) == 50.0

    def test_alpha_clamping(self):
        ema = ExponentialMovingAverage(alpha=1.5)
        assert ema.alpha == 1.0
        ema2 = ExponentialMovingAverage(alpha=-0.5)
        assert ema2.alpha == 0.0


class TestCoordinateSmoother:
    def test_first_value_passthrough(self):
        cs = CoordinateSmoother(alpha=0.5)
        x, y = cs.update(100.0, 200.0)
        assert x == 100.0
        assert y == 200.0

    def test_smoothing(self):
        cs = CoordinateSmoother(alpha=0.3)
        cs.update(0.0, 0.0)
        x, y = cs.update(100.0, 100.0)
        assert x < 50.0
        assert y < 50.0

    def test_reset(self):
        cs = CoordinateSmoother(alpha=0.5)
        cs.update(100.0, 200.0)
        cs.reset()
        x, y = cs.update(50.0, 50.0)
        assert x == 50.0
        assert y == 50.0


class TestAdaptiveCoordinateSmoother:
    def test_first_value_passthrough(self):
        acs = AdaptiveCoordinateSmoother()
        x, y = acs.update(100.0, 200.0)
        assert x == 100.0
        assert y == 200.0

    def test_slow_movement_heavy_smoothing(self):
        acs = AdaptiveCoordinateSmoother(alpha_slow=0.1, alpha_fast=0.9, velocity_threshold=100.0)
        acs.update(100.0, 100.0)
        x, y = acs.update(101.0, 101.0)  # Slow movement
        # Should be close to previous value (heavy smoothing)
        assert abs(x - 100.0) < 5.0

    def test_fast_movement_light_smoothing(self):
        acs = AdaptiveCoordinateSmoother(alpha_slow=0.1, alpha_fast=0.9, velocity_threshold=10.0)
        acs.update(0.0, 0.0)
        x, y = acs.update(100.0, 100.0)  # Fast movement (velocity > threshold)
        # Should be close to new value (light smoothing)
        assert x > 50.0

    def test_reset(self):
        acs = AdaptiveCoordinateSmoother()
        acs.update(100.0, 100.0)
        acs.reset()
        x, y = acs.update(50.0, 50.0)
        assert x == 50.0


class TestOneEuroFilter:
    def test_first_value_passthrough(self):
        f = OneEuroFilter()
        assert f.update(10.0, t=0.0) == 10.0

    def test_smoothing_noisy_signal(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        values = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        results = []
        for i, v in enumerate(values):
            results.append(f.update(v, t=i * 0.033))
        # Filtered signal should have smaller range than input
        assert max(results) - min(results) < 10.0

    def test_reset(self):
        f = OneEuroFilter()
        f.update(100.0, t=0.0)
        f.reset()
        assert f.update(50.0, t=1.0) == 50.0


class TestOneEuroFilter2D:
    def test_first_value(self):
        f = OneEuroFilter2D()
        x, y = f.update(100.0, 200.0, t=0.0)
        assert x == 100.0
        assert y == 200.0

    def test_reset(self):
        f = OneEuroFilter2D()
        f.update(100.0, 200.0, t=0.0)
        f.reset()
        x, y = f.update(50.0, 50.0, t=1.0)
        assert x == 50.0
        assert y == 50.0


class TestScreenMapper:
    def test_center_mapping(self):
        m = ScreenMapper(1920, 1080, flip_x=False, dead_zone=0)
        x, y = m.map_to_screen(0.5, 0.5)
        # Should be approximately center of screen
        assert abs(x - 960) < 50
        assert abs(y - 540) < 50

    def test_flip_x(self):
        m = ScreenMapper(1920, 1080, flip_x=True, dead_zone=0)
        x1, _ = m.map_to_screen(0.2, 0.5)
        m.reset()
        m_no_flip = ScreenMapper(1920, 1080, flip_x=False, dead_zone=0)
        x2, _ = m_no_flip.map_to_screen(0.2, 0.5)
        # Flipped x should map to opposite side
        assert x1 > x2

    def test_clamping(self):
        m = ScreenMapper(1920, 1080, flip_x=False, dead_zone=0)
        x, y = m.map_to_screen(-0.5, -0.5)
        assert x >= 0
        assert y >= 0
        x2, y2 = m.map_to_screen(1.5, 1.5)
        assert x2 <= 1919
        assert y2 <= 1079

    def test_dead_zone_suppresses_small_movement(self):
        m = ScreenMapper(1920, 1080, flip_x=False, dead_zone=10)
        x1, y1 = m.map_to_screen(0.5, 0.5)
        # Very small movement
        x2, y2 = m.map_to_screen(0.5001, 0.5001)
        # Should stay at same position due to dead zone
        assert x1 == x2
        assert y1 == y2

    def test_dead_zone_allows_large_movement(self):
        m = ScreenMapper(1920, 1080, flip_x=False, dead_zone=5)
        x1, y1 = m.map_to_screen(0.5, 0.5)
        # Large movement
        x2, y2 = m.map_to_screen(0.8, 0.8)
        assert x2 != x1 or y2 != y1

    def test_edge_proximity(self):
        m = ScreenMapper(1920, 1080)
        # Edge proximity at center should be 0
        assert m._edge_proximity(0.5, 0.5) == 0.0
        # Edge proximity at corners should be > 0
        assert m._edge_proximity(0.01, 0.01) > 0.0

    def test_reset(self):
        m = ScreenMapper(1920, 1080)
        m.map_to_screen(0.9, 0.9)
        m.reset()
        assert m._last_x == 960
        assert m._last_y == 540
