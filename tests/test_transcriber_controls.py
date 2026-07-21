import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transcriber import RealTimePipeline


class RealTimePipelineControlTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = RealTimePipeline.__new__(RealTimePipeline)
        pipeline.current_gender = "neutral"
        pipeline.current_age = "adult"
        pipeline.current_visual_mode = "asian_american"
        pipeline.current_prompt_style = "human_focus"
        pipeline.current_language = None
        pipeline.last_text = "a city glowing after rain"
        pipeline.last_prompt_token_count = 42
        pipeline.audio_status = "ready"
        pipeline.audio_reconnects = 0
        pipeline.last_audio_error = ""
        pipeline.state_lock = threading.Lock()
        pipeline.scene_lock = threading.Lock()
        pipeline.osc_lock = threading.Lock()
        pipeline.last_status_osc_time = 0.0
        pipeline.backend_retry_not_before = 0.0
        pipeline.backend_status = "ready"
        pipeline.backend = "faster_whisper"
        pipeline.is_speaking = False
        pipeline.last_total_latency = 0.25
        pipeline.last_inference_latency = 0.2
        pipeline.scheduler = Mock()
        pipeline.scheduler.metrics.return_value = SimpleNamespace(
            queue_depth=0,
            dropped_stale=0,
            dropped_finals=0,
        )
        pipeline.osc_client = Mock()
        return pipeline

    def test_visual_control_refreshes_the_current_prompt_immediately(self):
        pipeline = self.make_pipeline()
        pipeline.prompt_budgeter = None
        pipeline.build_visual_prompt = Mock(wraps=pipeline.build_visual_prompt)
        pipeline.send_runtime_status = Mock()

        with redirect_stdout(StringIO()):
            pipeline.apply_control("visual_mode", "black_brown")

        pipeline.build_visual_prompt.assert_called_once_with(
            "a city glowing after rain"
        )
        pipeline.osc_client.send_message.assert_any_call(
            "/control_ack",
            "visual_mode:black_brown",
        )
        prompt_messages = [
            call.args[1]
            for call in pipeline.osc_client.send_message.call_args_list
            if call.args[0] == "/prompt"
        ]
        self.assertEqual(len(prompt_messages), 1)
        self.assertIn("adult Black or Brown person", prompt_messages[0])
        pipeline.osc_client.send_message.assert_any_call("/prompt_tokens", 0)
        pipeline.send_runtime_status.assert_called_once_with(force=True)

    def test_language_control_does_not_rebuild_the_visual_prompt(self):
        pipeline = self.make_pipeline()
        pipeline.build_visual_prompt = Mock(return_value="unused")
        pipeline.send_runtime_status = Mock()

        with redirect_stdout(StringIO()):
            pipeline.apply_control("language", "es")

        pipeline.build_visual_prompt.assert_not_called()
        self.assertEqual(pipeline.current_language, "es")

    def test_runtime_status_includes_the_active_control_snapshot(self):
        pipeline = self.make_pipeline()
        pipeline.current_gender = "woman"
        pipeline.current_age = "elder"
        pipeline.current_visual_mode = "asian_black_brown"
        pipeline.current_prompt_style = "general_scene"
        pipeline.current_language = "zh"

        pipeline.send_runtime_status(force=True)

        expected = {
            ("/audio_status", "ready"),
            ("/audio_reconnects", 0),
            ("/audio_error", ""),
            ("/gender", "woman"),
            ("/age", "elder"),
            ("/visual_mode", "asian_black_brown"),
            ("/prompt_style", "general_scene"),
            ("/language", "zh"),
        }
        calls = {
            tuple(call.args)
            for call in pipeline.osc_client.send_message.call_args_list
        }
        self.assertTrue(expected.issubset(calls))


class MicrophoneRecoveryTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = RealTimePipeline.__new__(RealTimePipeline)
        pipeline.is_running = True
        pipeline.is_speaking = False
        pipeline.audio_status = "starting"
        pipeline.audio_reconnects = 0
        pipeline.last_audio_error = ""
        pipeline.send_runtime_status = Mock()
        pipeline.process_audio_data = Mock()
        pipeline.finalize_interrupted_audio = Mock()
        pipeline.wait_for_audio_retry = Mock(return_value=True)
        return pipeline

    @staticmethod
    def make_audio_interface():
        audio_interface = Mock()
        audio_interface.terminate = Mock()
        return audio_interface

    @staticmethod
    def make_stopping_stream(pipeline):
        stream = Mock()

        def read_once(*_args, **_kwargs):
            pipeline.is_running = False
            return b"\x00\x00" * 16

        stream.read.side_effect = read_once
        return stream

    def test_recovers_when_the_microphone_is_unavailable_at_startup(self):
        pipeline = self.make_pipeline()
        first_audio = self.make_audio_interface()
        second_audio = self.make_audio_interface()
        recovered_stream = self.make_stopping_stream(pipeline)
        pipeline.create_audio_interface = Mock(
            side_effect=[first_audio, second_audio]
        )
        pipeline.open_microphone_stream = Mock(
            side_effect=[OSError("device unavailable"), recovered_stream]
        )

        with redirect_stdout(StringIO()):
            pipeline.audio_callback()

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(pipeline.audio_status, "stopped")
        self.assertEqual(pipeline.last_audio_error, "")
        pipeline.process_audio_data.assert_called_once()
        first_audio.terminate.assert_called_once()
        second_audio.terminate.assert_called_once()
        recovered_stream.close.assert_called_once()

    def test_reopens_the_stream_after_repeated_read_failures(self):
        pipeline = self.make_pipeline()
        first_audio = self.make_audio_interface()
        second_audio = self.make_audio_interface()
        failing_stream = Mock()
        failing_stream.read.side_effect = [
            OSError("input overflow"),
            OSError("device disconnected"),
        ]
        recovered_stream = self.make_stopping_stream(pipeline)
        pipeline.create_audio_interface = Mock(
            side_effect=[first_audio, second_audio]
        )
        pipeline.open_microphone_stream = Mock(
            side_effect=[failing_stream, recovered_stream]
        )

        with (
            patch("transcriber.AUDIO_MAX_CONSECUTIVE_READ_ERRORS", 2),
            redirect_stdout(StringIO()),
        ):
            pipeline.audio_callback()

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(pipeline.audio_status, "stopped")
        pipeline.process_audio_data.assert_called_once()
        pipeline.finalize_interrupted_audio.assert_called_once()
        failing_stream.close.assert_called_once()
        recovered_stream.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
