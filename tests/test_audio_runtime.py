import unittest

import numpy as np

from audio_runtime import (
    EnergyVoiceActivityDetector,
    get_audio_input_device,
    list_audio_input_devices,
)


class FakeAudioInterface:
    def __init__(self):
        self.devices = [
            {
                "index": 0,
                "name": "Speakers",
                "maxInputChannels": 0,
                "defaultSampleRate": 48000,
            },
            {
                "index": 1,
                "name": "Built-in Microphone",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000,
            },
            {
                "index": 2,
                "name": "USB Microphone",
                "maxInputChannels": 1,
                "defaultSampleRate": 44100,
            },
        ]

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def get_default_input_device_info(self):
        return self.devices[1]


class EnergyVoiceActivityDetectorTests(unittest.TestCase):
    def test_detects_audio_above_the_energy_threshold(self):
        detector = EnergyVoiceActivityDetector(threshold=100)

        self.assertFalse(detector.is_speech(np.array([0, 50, -50], dtype=np.int16)))
        self.assertTrue(detector.is_speech(np.array([200, -200], dtype=np.int16)))


class AudioInputDeviceTests(unittest.TestCase):
    def test_lists_only_input_devices_and_marks_the_default(self):
        devices = list_audio_input_devices(FakeAudioInterface())

        self.assertEqual([device.index for device in devices], [1, 2])
        self.assertTrue(devices[0].is_default)
        self.assertFalse(devices[1].is_default)

    def test_resolves_an_explicit_input_device(self):
        device = get_audio_input_device(FakeAudioInterface(), 2)

        self.assertEqual(device.name, "USB Microphone")
        self.assertEqual(device.max_input_channels, 1)
        self.assertEqual(device.default_sample_rate, 44100.0)

    def test_rejects_an_output_only_device(self):
        with self.assertRaisesRegex(ValueError, "no input channels"):
            get_audio_input_device(FakeAudioInterface(), 0)


if __name__ == "__main__":
    unittest.main()
