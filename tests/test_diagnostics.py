import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diagnostics import _cuda_results, _microphone_result


class CudaDiagnosticTests(unittest.TestCase):
    @staticmethod
    def failing_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("CUDA runtime unavailable")
        return __import__(name, *args, **kwargs)

    def test_broken_torch_is_informational_for_online_backends(self):
        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch("builtins.__import__", side_effect=self.failing_torch_import),
        ):
            results = _cuda_results("cuda", required=False)

        self.assertEqual(results[0].status, "INFO")

    def test_broken_torch_fails_a_local_backend(self):
        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch("builtins.__import__", side_effect=self.failing_torch_import),
        ):
            results = _cuda_results("cuda", required=True)

        self.assertEqual(results[0].status, "FAIL")


class MicrophoneDiagnosticTests(unittest.TestCase):
    def test_checks_the_configured_input_device(self):
        audio_interface = Mock()
        stream = Mock()
        audio_interface.open.return_value = stream
        pyaudio_module = types.ModuleType("pyaudio")
        pyaudio_module.paInt16 = 8
        pyaudio_module.PyAudio = Mock(return_value=audio_interface)
        device = SimpleNamespace(index=7, name="Installation Microphone")

        with (
            patch.dict(sys.modules, {"pyaudio": pyaudio_module}),
            patch("diagnostics.get_audio_input_device", return_value=device) as resolve,
        ):
            result = _microphone_result(16000, 1024, input_device_index=7)

        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.detail, "[7] Installation Microphone")
        resolve.assert_called_once_with(audio_interface, 7)
        self.assertEqual(audio_interface.open.call_args.kwargs["input_device_index"], 7)
        stream.close.assert_called_once()
        audio_interface.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
