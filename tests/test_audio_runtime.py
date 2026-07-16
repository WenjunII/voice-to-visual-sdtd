import unittest

import numpy as np

from audio_runtime import EnergyVoiceActivityDetector


class EnergyVoiceActivityDetectorTests(unittest.TestCase):
    def test_detects_audio_above_the_energy_threshold(self):
        detector = EnergyVoiceActivityDetector(threshold=100)

        self.assertFalse(detector.is_speech(np.array([0, 50, -50], dtype=np.int16)))
        self.assertTrue(detector.is_speech(np.array([200, -200], dtype=np.int16)))


if __name__ == "__main__":
    unittest.main()
