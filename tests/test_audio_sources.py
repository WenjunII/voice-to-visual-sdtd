import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audio_sources import (
    AudioSourceFinished,
    AudioSourceStopped,
    PyAudioSource,
    WavReplaySource,
    list_system_audio_input_devices,
    load_wav_samples,
)


def write_wav(path, samples, *, sample_rate=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)


class PyAudioSourceTests(unittest.TestCase):
    @staticmethod
    def make_module():
        stream = Mock()
        stream.read.return_value = b"\x01\x00" * 8
        audio_interface = Mock()
        audio_interface.get_default_input_device_info.return_value = {
            "index": 3,
            "name": "USB Microphone",
            "maxInputChannels": 1,
            "defaultSampleRate": 48000.0,
        }
        audio_interface.get_device_info_by_index.return_value = (
            audio_interface.get_default_input_device_info.return_value
        )
        audio_interface.open.return_value = stream
        module = SimpleNamespace(
            paInt16=8,
            PyAudio=Mock(return_value=audio_interface),
        )
        return module, audio_interface, stream

    def test_owns_microphone_open_read_and_close(self):
        module, audio_interface, stream = self.make_module()
        source = PyAudioSource(
            device_index=3,
            sample_rate=16000,
            chunk_samples=1024,
            pyaudio_module=module,
        )

        source.open()
        data = source.read()
        source.close()

        self.assertEqual(data, b"\x01\x00" * 8)
        self.assertEqual(source.device_index, 3)
        self.assertEqual(source.name, "USB Microphone")
        self.assertEqual(
            audio_interface.open.call_args.kwargs,
            {
                "format": 8,
                "channels": 1,
                "rate": 16000,
                "input": True,
                "input_device_index": 3,
                "frames_per_buffer": 1024,
            },
        )
        stream.read.assert_called_once_with(
            1024,
            exception_on_overflow=False,
        )
        stream.stop_stream.assert_called_once_with()
        stream.close.assert_called_once_with()
        audio_interface.terminate.assert_called_once_with()

    def test_failed_open_terminates_the_audio_interface(self):
        module, audio_interface, _stream = self.make_module()
        audio_interface.open.side_effect = OSError("device busy")
        source = PyAudioSource(pyaudio_module=module)

        with self.assertRaisesRegex(OSError, "device busy"):
            source.open()

        audio_interface.terminate.assert_called_once_with()

    def test_rejects_invalid_capture_dimensions(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PyAudioSource(sample_rate=0)

    def test_missing_pyaudio_error_points_to_diagnostics(self):
        source = PyAudioSource()
        with patch(
            "audio_sources._load_pyaudio",
            side_effect=RuntimeError("Run --diagnose"),
        ):
            with self.assertRaisesRegex(RuntimeError, "--diagnose"):
                source.open()

    def test_device_listing_terminates_its_interface(self):
        module, audio_interface, _stream = self.make_module()
        audio_interface.get_device_count.return_value = 1

        devices = list_system_audio_input_devices(module)

        self.assertEqual([device.name for device in devices], ["USB Microphone"])
        audio_interface.terminate.assert_called_once_with()


class WavReplaySourceTests(unittest.TestCase):
    def test_decodes_stereo_and_resamples_to_the_runtime_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            stereo = np.array(
                [[1000, 3000], [-1000, 1000], [2000, 4000]],
                dtype="<i2",
            )
            write_wav(
                path,
                stereo.tobytes(),
                sample_rate=8000,
                channels=2,
            )

            samples = load_wav_samples(path, target_sample_rate=16000)

        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(len(samples), 6)
        self.assertEqual(samples[0], 2000)
        self.assertEqual(samples[-1], 3000)

    def test_replays_stable_chunks_and_signals_end_of_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.wav"
            original = np.arange(10, dtype="<i2")
            write_wav(path, original.tobytes())
            source = WavReplaySource(
                path,
                chunk_samples=4,
                realtime=False,
            )

            source.open()
            chunks = [source.read(), source.read(), source.read()]
            with self.assertRaises(AudioSourceFinished):
                source.read()
            source.close()

        replayed = np.frombuffer(b"".join(chunks), dtype="<i2")
        np.testing.assert_array_equal(replayed, original)

    def test_realtime_wait_is_cooperatively_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancel.wav"
            write_wav(path, np.ones(1600, dtype="<i2").tobytes())
            stop_event = threading.Event()
            stop_event.set()
            source = WavReplaySource(
                path,
                chunk_samples=1600,
                realtime=True,
                stop_event=stop_event,
            )
            source.open()

            with self.assertRaises(AudioSourceStopped):
                source.read()

    def test_rejects_non_16_bit_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eight-bit.wav"
            write_wav(path, bytes([128] * 16), width=1)

            with self.assertRaisesRegex(ValueError, "16-bit PCM"):
                WavReplaySource(path, realtime=False)

    def test_rejects_invalid_runtime_format(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            WavReplaySource("unused.wav", chunk_samples=0)


if __name__ == "__main__":
    unittest.main()
