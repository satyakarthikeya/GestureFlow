
import pytest
from unittest.mock import MagicMock
import sys

# Mock all dependencies of GestureClassifier
sys.modules['core.landmark_utils'] = MagicMock()
sys.modules['core.hand_tracker'] = MagicMock()
sys.modules['core.smoothing'] = MagicMock()
sys.modules['core.coordinate_mapper'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pyautogui'] = MagicMock()
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()

# Specifically mock LandmarkUtils and its methods
class MockLandmarkUtils:
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    FINGER_NAMES = ["index", "middle", "ring", "pinky"]

    @staticmethod
    def is_finger_extended(landmarks, finger_idx, threshold=0.03):
        tip = MockLandmarkUtils.FINGER_TIPS[finger_idx]
        pip = MockLandmarkUtils.FINGER_PIPS[finger_idx]
        return landmarks[tip][1] < landmarks[pip][1] - threshold

    @staticmethod
    def is_thumb_extended(landmarks, handedness="Right"):
        return landmarks[4][0] < landmarks[3][0] if handedness == "Right" else landmarks[4][0] > landmarks[3][0]

    @staticmethod
    def get_all_finger_states(landmarks, handedness="Right", threshold=0.03):
        states = {"thumb": MockLandmarkUtils.is_thumb_extended(landmarks, handedness)}
        for i, name in enumerate(MockLandmarkUtils.FINGER_NAMES):
            states[name] = MockLandmarkUtils.is_finger_extended(landmarks, i, threshold)
        return states

sys.modules['core.landmark_utils'].LandmarkUtils = MockLandmarkUtils

# Mock the controllers
sys.modules['apps.hci.controllers.brightness'] = MagicMock()
sys.modules['apps.hci.controllers.cursor'] = MagicMock()
sys.modules['apps.hci.controllers.media'] = MagicMock()
sys.modules['apps.hci.controllers.scroll'] = MagicMock()
sys.modules['apps.hci.controllers.tab_switch'] = MagicMock()
sys.modules['apps.hci.controllers.volume'] = MagicMock()

from apps.hci.gesture_classifier import GestureClassifier

def _make_landmarks():
    return [(0.5, 0.5, 0.0)] * 21

def get_fist_landmarks():
    lms = _make_landmarks()
    for i in range(4):
        lms[MockLandmarkUtils.FINGER_PIPS[i]] = (0.5, 0.4, 0.0)
        lms[MockLandmarkUtils.FINGER_TIPS[i]] = (0.5, 0.55, 0.0)
    lms[3] = (0.40, 0.40, 0.0); lms[4] = (0.41, 0.41, 0.0)
    return lms

def get_rock_on_landmarks():
    lms = get_fist_landmarks()
    lms[MockLandmarkUtils.FINGER_PIPS[0]] = (0.5, 0.5, 0.0); lms[MockLandmarkUtils.FINGER_TIPS[0]] = (0.5, 0.3, 0.0)
    lms[MockLandmarkUtils.FINGER_PIPS[3]] = (0.5, 0.5, 0.0); lms[MockLandmarkUtils.FINGER_TIPS[3]] = (0.5, 0.3, 0.0)
    return lms

def get_thumbs_up_landmarks():
    lms = get_fist_landmarks()
    lms[MockLandmarkUtils.THUMB_MCP] = (0.45, 0.45, 0.0)
    lms[MockLandmarkUtils.THUMB_IP] = (0.40, 0.38, 0.0)
    lms[MockLandmarkUtils.THUMB_TIP] = (0.35, 0.25, 0.0)
    return lms

def get_thumbs_down_landmarks():
    lms = get_fist_landmarks()
    lms[MockLandmarkUtils.THUMB_MCP] = (0.45, 0.45, 0.0)
    lms[MockLandmarkUtils.THUMB_IP] = (0.40, 0.52, 0.0)
    lms[MockLandmarkUtils.THUMB_TIP] = (0.35, 0.65, 0.0)
    return lms

def get_open_hand_landmarks():
    lms = _make_landmarks()
    for i in range(4):
        lms[MockLandmarkUtils.FINGER_PIPS[i]] = (0.5, 0.5, 0.0)
        lms[MockLandmarkUtils.FINGER_TIPS[i]] = (0.5, 0.3, 0.0)
    lms[3] = (0.40, 0.40, 0.0); lms[4] = (0.35, 0.35, 0.0)
    return lms

def get_count_landmarks(count):
    lms = get_fist_landmarks()
    for i in range(count):
        lms[MockLandmarkUtils.FINGER_PIPS[i]] = (0.5, 0.5, 0.0)
        lms[MockLandmarkUtils.FINGER_TIPS[i]] = (0.5, 0.3, 0.0)
    return lms

class TestGestureClassifierRefactored:
    def test_fist(self):
        classifier = GestureClassifier()
        lms = get_fist_landmarks()
        assert classifier._identify_gesture(lms, "Right") == "fist"

    def test_rock_on(self):
        classifier = GestureClassifier()
        lms = get_rock_on_landmarks()
        assert classifier._identify_gesture(lms, "Right") == "rock_on"

    def test_thumbs_up(self):
        classifier = GestureClassifier()
        lms = get_thumbs_up_landmarks()
        assert classifier._identify_gesture(lms, "Right") == "thumbs_up"

    def test_thumbs_down(self):
        classifier = GestureClassifier()
        lms = get_thumbs_down_landmarks()
        assert classifier._identify_gesture(lms, "Right") == "thumbs_down"

    def test_one_finger(self):
        classifier = GestureClassifier()
        lms = get_count_landmarks(1)
        assert classifier._identify_gesture(lms, "Right") == "one_finger"

    def test_two_fingers(self):
        classifier = GestureClassifier()
        lms = get_count_landmarks(2)
        assert classifier._identify_gesture(lms, "Right") == "two_fingers"

    def test_three_fingers(self):
        classifier = GestureClassifier()
        lms = get_count_landmarks(3)
        assert classifier._identify_gesture(lms, "Right") == "three_fingers"

    def test_four_fingers(self):
        classifier = GestureClassifier()
        lms = get_count_landmarks(4)
        assert classifier._identify_gesture(lms, "Right") == "four_fingers"

    def test_open_hand(self):
        classifier = GestureClassifier()
        lms = get_open_hand_landmarks()
        assert classifier._identify_gesture(lms, "Right") == "open_hand"
