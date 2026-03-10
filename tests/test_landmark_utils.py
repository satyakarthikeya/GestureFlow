"""Tests for core.landmark_utils — finger detection, gestures, distances."""

import math
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

import numpy as np

import pytest

from core.landmark_utils import LandmarkUtils


def _make_landmarks(overrides=None):
    """Create a default set of 21 landmarks (all at center, fingers curled).

    Override specific indices with (x, y, z) tuples.
    """
    # Default: all landmarks at (0.5, 0.5, 0.0)
    landmarks = [(0.5, 0.5, 0.0)] * 21
    if overrides:
        for idx, val in overrides.items():
            landmarks[idx] = val
    return landmarks


def _open_hand_landmarks(handedness="Right"):
    """All fingers extended (tips above PIPs)."""
    lms = list(_make_landmarks())
    # Thumb: tip far from MCP, and positioned correctly for handedness
    lms[1] = (0.5, 0.5, 0.0)    # THUMB_CMC
    lms[2] = (0.45, 0.45, 0.0)  # THUMB_MCP
    lms[3] = (0.40, 0.40, 0.0)  # THUMB_IP
    if handedness == "Right":
        lms[4] = (0.35, 0.35, 0.0)  # THUMB_TIP — left of IP for Right hand
    else:
        lms[4] = (0.55, 0.35, 0.0)  # THUMB_TIP — right of IP for Left hand

    # Index: tip above PIP
    lms[6] = (0.4, 0.5, 0.0)    # INDEX_PIP
    lms[8] = (0.4, 0.3, 0.0)    # INDEX_TIP (above PIP)

    # Middle: tip above PIP
    lms[10] = (0.5, 0.5, 0.0)   # MIDDLE_PIP
    lms[12] = (0.5, 0.3, 0.0)   # MIDDLE_TIP

    # Ring: tip above PIP
    lms[14] = (0.6, 0.5, 0.0)   # RING_PIP
    lms[16] = (0.6, 0.3, 0.0)   # RING_TIP

    # Pinky: tip above PIP
    lms[18] = (0.7, 0.5, 0.0)   # PINKY_PIP
    lms[20] = (0.7, 0.3, 0.0)   # PINKY_TIP

    return lms


def _fist_landmarks(handedness="Right"):
    """All fingers curled (tips below PIPs)."""
    lms = list(_make_landmarks())
    # Thumb: tip close to MCP
    lms[2] = (0.45, 0.45, 0.0)  # THUMB_MCP
    lms[3] = (0.46, 0.46, 0.0)  # THUMB_IP
    lms[4] = (0.47, 0.48, 0.0)  # THUMB_TIP (close to MCP, below IP)

    # All fingertips BELOW their PIP joints
    lms[6] = (0.4, 0.4, 0.0)    # INDEX_PIP
    lms[8] = (0.4, 0.55, 0.0)   # INDEX_TIP (below PIP)
    lms[10] = (0.5, 0.4, 0.0)   # MIDDLE_PIP
    lms[12] = (0.5, 0.55, 0.0)  # MIDDLE_TIP
    lms[14] = (0.6, 0.4, 0.0)   # RING_PIP
    lms[16] = (0.6, 0.55, 0.0)  # RING_TIP
    lms[18] = (0.7, 0.4, 0.0)   # PINKY_PIP
    lms[20] = (0.7, 0.55, 0.0)  # PINKY_TIP

    return lms


class TestFingerExtension:
    def test_index_extended(self):
        lms = _open_hand_landmarks()
        assert LandmarkUtils.is_finger_extended(lms, 0) is True

    def test_index_curled(self):
        lms = _fist_landmarks()
        assert LandmarkUtils.is_finger_extended(lms, 0) is False

    def test_all_fingers_extended(self):
        lms = _open_hand_landmarks()
        for i in range(4):
            assert LandmarkUtils.is_finger_extended(lms, i) is True

    def test_all_fingers_curled(self):
        lms = _fist_landmarks()
        for i in range(4):
            assert LandmarkUtils.is_finger_extended(lms, i) is False

    def test_custom_threshold(self):
        lms = list(_make_landmarks())
        lms[6] = (0.5, 0.50, 0.0)   # INDEX_PIP
        lms[8] = (0.5, 0.48, 0.0)   # INDEX_TIP (barely above PIP)
        # With default threshold (0.03), this should NOT be extended
        assert LandmarkUtils.is_finger_extended(lms, 0, threshold=0.03) is False
        # With very small threshold, it should be extended
        assert LandmarkUtils.is_finger_extended(lms, 0, threshold=0.01) is True


class TestThumbExtension:
    def test_thumb_extended_right(self):
        lms = _open_hand_landmarks("Right")
        assert LandmarkUtils.is_thumb_extended(lms, "Right") is True

    def test_thumb_extended_left(self):
        lms = _open_hand_landmarks("Left")
        assert LandmarkUtils.is_thumb_extended(lms, "Left") is True

    def test_thumb_curled_right(self):
        lms = _fist_landmarks("Right")
        assert LandmarkUtils.is_thumb_extended(lms, "Right") is False


class TestFingerStates:
    def test_open_hand_all_extended(self):
        lms = _open_hand_landmarks()
        states = LandmarkUtils.get_all_finger_states(lms, "Right")
        assert all(states.values())

    def test_fist_all_curled(self):
        lms = _fist_landmarks()
        states = LandmarkUtils.get_all_finger_states(lms, "Right")
        assert not any(states.values())

    def test_count_open_hand(self):
        lms = _open_hand_landmarks()
        assert LandmarkUtils.count_extended_fingers(lms, "Right") == 5

    def test_count_fist(self):
        lms = _fist_landmarks()
        assert LandmarkUtils.count_extended_fingers(lms, "Right") == 0

    def test_extended_names_open_hand(self):
        lms = _open_hand_landmarks()
        names = LandmarkUtils.get_extended_finger_names(lms, "Right")
        assert set(names) == {"thumb", "index", "middle", "ring", "pinky"}


class TestPinchDistance:
    def test_pinch_close(self):
        lms = list(_make_landmarks())
        lms[4] = (0.5, 0.5, 0.0)   # THUMB_TIP
        lms[8] = (0.51, 0.5, 0.0)  # INDEX_TIP (very close)
        dist = LandmarkUtils.pinch_distance(lms)
        assert dist < 0.02

    def test_pinch_far(self):
        lms = list(_make_landmarks())
        lms[4] = (0.3, 0.3, 0.0)  # THUMB_TIP
        lms[8] = (0.7, 0.7, 0.0)  # INDEX_TIP (far)
        dist = LandmarkUtils.pinch_distance(lms)
        assert dist > 0.4


class TestLandmarkDistance:
    def test_same_point(self):
        lms = _make_landmarks()
        assert LandmarkUtils.landmark_distance(lms, 0, 1) == 0.0

    def test_known_distance(self):
        lms = list(_make_landmarks())
        lms[0] = (0.0, 0.0, 0.0)
        lms[1] = (0.3, 0.4, 0.0)
        dist = LandmarkUtils.landmark_distance(lms, 0, 1)
        assert abs(dist - 0.5) < 1e-6


class TestPalmCenter:
    def test_center_computation(self):
        lms = list(_make_landmarks())
        lms[0] = (0.0, 0.0, 0.0)   # WRIST
        lms[5] = (0.2, 0.2, 0.0)   # INDEX_MCP
        lms[9] = (0.4, 0.4, 0.0)   # MIDDLE_MCP
        lms[13] = (0.6, 0.6, 0.0)  # RING_MCP
        lms[17] = (0.8, 0.8, 0.0)  # PINKY_MCP
        cx, cy = LandmarkUtils.palm_center(lms)
        assert abs(cx - 0.4) < 1e-6
        assert abs(cy - 0.4) < 1e-6


class TestHandVelocity:
    def test_stationary(self):
        vx, vy = LandmarkUtils.hand_velocity((0.5, 0.5), (0.5, 0.5), 1.0)
        assert vx == 0.0
        assert vy == 0.0

    def test_moving_right(self):
        vx, vy = LandmarkUtils.hand_velocity((0.6, 0.5), (0.5, 0.5), 0.5)
        assert vx > 0

    def test_zero_dt(self):
        vx, vy = LandmarkUtils.hand_velocity((0.6, 0.5), (0.5, 0.5), 0.0)
        assert vx == 0.0 and vy == 0.0


class TestGestureDetection:
    def test_is_fist(self):
        lms = _fist_landmarks()
        assert LandmarkUtils.is_fist(lms) is True

    def test_open_hand_not_fist(self):
        lms = _open_hand_landmarks()
        assert LandmarkUtils.is_fist(lms) is False

    def test_rock_on(self):
        lms = _fist_landmarks()
        # Extend index and pinky only
        lms[6] = (0.4, 0.5, 0.0)   # INDEX_PIP
        lms[8] = (0.4, 0.3, 0.0)   # INDEX_TIP (extended)
        lms[18] = (0.7, 0.5, 0.0)  # PINKY_PIP
        lms[20] = (0.7, 0.3, 0.0)  # PINKY_TIP (extended)
        assert LandmarkUtils.is_rock_on(lms) is True

    def test_thumbs_up(self):
        lms = _fist_landmarks()
        # Extend thumb upward
        lms[2] = (0.45, 0.45, 0.0)  # THUMB_MCP
        lms[3] = (0.40, 0.38, 0.0)  # THUMB_IP
        lms[4] = (0.35, 0.25, 0.0)  # THUMB_TIP (way above MCP, far from MCP for extended)
        assert LandmarkUtils.is_thumbs_up(lms, "Right") is True

    def test_thumbs_down(self):
        lms = _fist_landmarks()
        # Extend thumb downward
        lms[2] = (0.45, 0.45, 0.0)  # THUMB_MCP
        lms[3] = (0.40, 0.52, 0.0)  # THUMB_IP
        lms[4] = (0.35, 0.60, 0.0)  # THUMB_TIP (way below MCP, far from MCP for extended)
        assert LandmarkUtils.is_thumbs_down(lms, "Right") is True


class TestNormalizeToScreen:
    def test_center(self):
        x, y = LandmarkUtils.normalize_to_screen(0.5, 0.5, 1920, 1080)
        assert x == 960
        assert y == 540

    def test_clamp_negative(self):
        x, y = LandmarkUtils.normalize_to_screen(-0.1, -0.1, 1920, 1080)
        assert x == 0
        assert y == 0

    def test_clamp_overflow(self):
        x, y = LandmarkUtils.normalize_to_screen(1.5, 1.5, 1920, 1080)
        assert x == 1919
        assert y == 1079
