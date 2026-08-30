import sys
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diagnostics import (
    _cuda_results,
    _microphone_result,
    _package_results,
    _prompt_budget_result,
)
from runtime_config import RuntimeConfig


class CudaDiagnosticTests(unittest.TestCase):
    @staticmethod
    def failing_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("CUDA runtime unavailable")
        return __import__(name, *args, **kwargs)

    def test_broken_torch_is_informational_for_online_backends(self):
        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch("diagnostics.metadata.version", return_value="2.7.0"),
            patch("builtins.__import__", side_effect=self.failing_torch_import) as importer,
        ):
            results = _cuda_results("cuda", required=False)

        self.assertEqual(results[0].status, "INFO")
        self.assertIn("not loaded", results[0].detail)
        self.assertEqual(results[1].status, "INFO")
        importer.assert_not_called()

    def test_broken_torch_fails_a_local_backend(self):
        with (
            patch("diagnostics.importlib.util.find_spec", return_value=object()),
            patch("builtins.__import__", side_effect=self.failing_torch_import),
        ):
            results = _cuda_results("cuda", required=True)

        self.assertEqual(results[0].status, "FAIL")

    def test_missing_torch_names_the_local_backend_profile(self):
        command = (
            "python -m pip install -r requirements/faster-whisper.txt"
        )
        with patch(
            "diagnostics.importlib.util.find_spec",
            return_value=None,
        ):
            results = _cuda_results(
                "cuda",
                required=True,
                install_command=command,
            )

        self.assertEqual(results[0].status, "FAIL")
        self.assertIn(command, results[0].detail)


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


class PackageDiagnosticTests(unittest.TestCase):
    def test_missing_required_package_names_the_backend_profile(self):
        with patch(
            "diagnostics.importlib.util.find_spec",
            return_value=None,
        ):
            results = _package_results("google")

        speech_recognition = next(
            result
            for result in results
            if result.name == "SpeechRecognition"
        )
        self.assertEqual(speech_recognition.status, "FAIL")
        self.assertIn(
            "python -m pip install -r requirements/google.txt",
            speech_recognition.detail,
        )
        argos = next(
            result for result in results if result.name == "Argos Translate"
        )
        self.assertEqual(argos.status, "INFO")
        self.assertEqual(argos.detail, "not installed (optional)")


class PromptBudgetDiagnosticTests(unittest.TestCase):
    def test_reports_exact_budgeting_when_every_tokenizer_loads_locally(self):
        loader = Mock()
        config = replace(
            RuntimeConfig(),
            prompt_tokenizer_models=("tokenizer-a", "tokenizer-b"),
        )

        result = _prompt_budget_result(config, tokenizer_loader=loader)

        self.assertEqual(result.status, "PASS")
        self.assertIn("exact budgeting ready", result.detail)
        self.assertEqual(
            [call.args for call in loader.call_args_list],
            [("tokenizer-a",), ("tokenizer-b",)],
        )
        self.assertTrue(
            all(
                call.kwargs == {"local_files_only": True}
                for call in loader.call_args_list
            )
        )

    def test_reports_conservative_fallback_when_cache_is_incomplete(self):
        loader = Mock(side_effect=[Mock(), OSError("not cached")])
        config = replace(
            RuntimeConfig(),
            prompt_tokenizer_models=("tokenizer-a", "tokenizer-b"),
            prompt_token_budget_fallback="conservative",
        )

        result = _prompt_budget_result(config, tokenizer_loader=loader)

        self.assertEqual(result.status, "WARN")
        self.assertIn("1/2 available locally", result.detail)
        self.assertIn("conservative offline budgeting", result.detail)

    def test_missing_cache_is_a_failure_when_fallback_is_off(self):
        config = replace(
            RuntimeConfig(),
            prompt_tokenizer_models=("tokenizer-a",),
            prompt_token_budget_fallback="off",
        )

        result = _prompt_budget_result(
            config,
            tokenizer_loader=Mock(side_effect=OSError("not cached")),
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIn("fallback is off", result.detail)

    def test_disabled_budgeting_does_not_load_tokenizers(self):
        loader = Mock()
        config = replace(
            RuntimeConfig(),
            prompt_token_budget_enabled=False,
        )

        result = _prompt_budget_result(config, tokenizer_loader=loader)

        self.assertEqual(result.status, "INFO")
        self.assertEqual(result.detail, "budgeting disabled")
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
