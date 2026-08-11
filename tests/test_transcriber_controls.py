import sys
import threading
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend_errors import RetryableTranscriptionError
from runtime_config import RuntimeConfig
from transcriber import RealTimePipeline, parse_args


def make_backend_adapter(*, online=False):
    adapter = Mock()
    adapter.name = "google" if online else "whisper"
    adapter.online = online
    adapter.request_interval = 2.5
    adapter.minimum_audio_seconds = 1.25
    adapter.maximum_audio_seconds = 5.0
    adapter.transcribe.return_value = "adapter transcript"
    return adapter


class TestLogSession:
    path = None

    def __init__(self):
        self.loggers = {}

    def logger(self, subsystem):
        return self.loggers.setdefault(subsystem, Mock())

    def close(self):
        return None


def make_pipeline(config=None, *, online=False, backend_adapter=None):
    return RealTimePipeline(
        enable_vad=False,
        enable_osc=False,
        enable_prompt_budget=False,
        enable_osc_controls=False,
        config=config if config is not None else RuntimeConfig(),
        backend_adapter=(
            backend_adapter
            if backend_adapter is not None
            else make_backend_adapter(online=online)
        ),
        log_session=TestLogSession(),
    )


class CommandLineTests(unittest.TestCase):
    def test_accepts_the_config_check_flag(self):
        args = parse_args(["--check-config"])

        self.assertTrue(args.check_config)

    def test_accepts_an_audio_input_device_override(self):
        args = parse_args(["--input-device", "4"])

        self.assertEqual(args.input_device, 4)

    def test_rejects_a_negative_audio_input_device(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--input-device", "-1"])


class RealTimePipelineControlTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = make_pipeline()
        pipeline.last_text = "a city glowing after rain"
        pipeline.last_prompt_token_count = 42
        pipeline.audio_status = "ready"
        pipeline.audio_device_index = 2
        pipeline.audio_device_name = "USB Microphone"
        pipeline.last_total_latency = 0.25
        pipeline.last_inference_latency = 0.2
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
            ("/audio_device_index", 2),
            ("/audio_device_name", "USB Microphone"),
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


class BackendAdapterIntegrationTests(unittest.TestCase):
    def make_pipeline(self, *, online=True):
        adapter = make_backend_adapter(online=online)
        pipeline = make_pipeline(
            online=online,
            backend_adapter=adapter,
        )
        return pipeline, adapter

    def test_delegates_transcription_and_language_to_the_adapter(self):
        pipeline, adapter = self.make_pipeline()
        pipeline.current_language = "es"
        samples = object()

        text = pipeline.transcribe_audio(samples)

        self.assertEqual(text, "adapter transcript")
        adapter.transcribe.assert_called_once_with(
            samples,
            language="es",
        )

    def test_uses_adapter_audio_and_timing_limits(self):
        pipeline, _adapter = self.make_pipeline()

        self.assertEqual(pipeline.minimum_audio_seconds(), 1.25)
        self.assertEqual(pipeline.maximum_audio_seconds(), 5.0)
        self.assertEqual(pipeline.online_request_interval(), 2.5)
        self.assertEqual(pipeline.local_request_interval(), 0.0)

    def test_closes_the_backend_adapter(self):
        pipeline, adapter = self.make_pipeline()

        pipeline.close()
        pipeline.close()

        adapter.close.assert_called_once_with()


class PipelineLifecycleTests(unittest.TestCase):
    def test_shutdown_interrupts_an_audio_retry_wait(self):
        pipeline = make_pipeline()
        result = []
        waiter = threading.Thread(
            target=lambda: result.append(
                pipeline.wait_for_audio_retry(10.0)
            )
        )
        waiter.start()

        pipeline.request_shutdown("test")
        waiter.join(timeout=0.5)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])
        pipeline.close()

    def test_close_waits_for_workers_before_closing_resources(self):
        pipeline = make_pipeline(
            replace(
                RuntimeConfig(),
                runtime_shutdown_grace_seconds=0.02,
            )
        )
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_stopped = threading.Event()
        backend_closed_after_worker = []

        def blocked_audio_worker():
            worker_started.set()
            release_worker.wait()
            worker_stopped.set()

        pipeline.audio_callback = blocked_audio_worker
        pipeline.transcription_loop = Mock()
        pipeline.backend_adapter.close.side_effect = lambda: (
            backend_closed_after_worker.append(worker_stopped.is_set())
        )
        workers = pipeline.start_worker_threads()
        self.assertTrue(worker_started.wait(timeout=0.5))
        self.assertTrue(all(not worker.daemon for worker in workers))

        release_timer = threading.Timer(0.08, release_worker.set)
        release_timer.start()
        try:
            pipeline.close()
        finally:
            release_worker.set()
            release_timer.join(timeout=0.5)

        self.assertTrue(worker_stopped.is_set())
        self.assertEqual(backend_closed_after_worker, [True])
        warning = pipeline.runtime_logger.warning.call_args
        self.assertEqual(
            warning.kwargs["extra"]["event"],
            "worker_shutdown_overdue",
        )
        self.assertEqual(
            warning.kwargs["extra"]["workers"],
            ["voice-to-visual-audio"],
        )

    def test_worker_crash_requests_pipeline_shutdown(self):
        pipeline = make_pipeline()

        def crashing_audio_worker():
            raise RuntimeError("audio worker failed")

        pipeline.audio_callback = crashing_audio_worker
        pipeline.transcription_loop = lambda: pipeline.stop_event.wait(0.5)
        pipeline.start_worker_threads()
        pipeline.join_worker_threads()

        self.assertTrue(pipeline.stop_event.is_set())
        crash = pipeline.runtime_logger.exception.call_args
        self.assertEqual(crash.kwargs["extra"]["event"], "worker_crashed")
        self.assertEqual(crash.kwargs["extra"]["worker"], "audio")
        pipeline.close()


class PipelineConfigurationIsolationTests(unittest.TestCase):
    def test_each_pipeline_configures_its_own_runtime_components(self):
        first_config = replace(
            RuntimeConfig(),
            audio_input_device_index=2,
            scene_memory_max_words=12,
            scene_memory_max_age_seconds=8.0,
            transcription_max_final_jobs=3,
            transcription_partial_max_age_seconds=1.5,
            transcription_final_max_age_seconds=9.0,
            vad_pre_roll_seconds=0.1,
            vad_silence_seconds=0.25,
            stream_overlap_seconds=0.2,
        )
        second_config = replace(
            RuntimeConfig(),
            audio_input_device_index=7,
            scene_memory_max_words=48,
            scene_memory_max_age_seconds=30.0,
            transcription_max_final_jobs=11,
            transcription_partial_max_age_seconds=5.0,
            transcription_final_max_age_seconds=45.0,
            vad_pre_roll_seconds=0.6,
            vad_silence_seconds=1.1,
            stream_overlap_seconds=0.8,
        )

        first = make_pipeline(first_config)
        second = make_pipeline(second_config)

        self.assertIs(first.config, first_config)
        self.assertIs(second.config, second_config)
        self.assertEqual(first.audio_device_index, 2)
        self.assertEqual(second.audio_device_index, 7)
        self.assertEqual(first.scheduler.max_final_jobs, 3)
        self.assertEqual(second.scheduler.max_final_jobs, 11)
        self.assertEqual(first.scheduler.partial_max_age_seconds, 1.5)
        self.assertEqual(second.scheduler.final_max_age_seconds, 45.0)
        self.assertEqual(first.scene_memory.max_words, 12)
        self.assertEqual(second.scene_memory.max_words, 48)
        self.assertEqual(first.segmenter.end_silence_samples, 4000)
        self.assertEqual(second.segmenter.end_silence_samples, 17600)
        self.assertEqual(first.segmenter.overlap_chunks, 4)
        self.assertEqual(second.segmenter.overlap_chunks, 13)

    def test_each_pipeline_uses_its_own_osc_configuration(self):
        first_config = replace(
            RuntimeConfig(),
            osc_ip="127.0.0.2",
            osc_port=7100,
            osc_control_enabled=True,
            osc_control_ip="127.0.0.4",
            osc_control_port=7101,
        )
        second_config = replace(
            RuntimeConfig(),
            osc_ip="127.0.0.3",
            osc_port=7200,
            osc_control_enabled=False,
        )

        with patch("transcriber.udp_client.SimpleUDPClient") as client:
            first = RealTimePipeline(
                enable_vad=False,
                enable_osc=True,
                enable_prompt_budget=False,
                enable_osc_controls=True,
                config=first_config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )
            second = RealTimePipeline(
                enable_vad=False,
                enable_osc=True,
                enable_prompt_budget=False,
                enable_osc_controls=True,
                config=second_config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        self.assertEqual(
            [call.args for call in client.call_args_list],
            [("127.0.0.2", 7100), ("127.0.0.3", 7200)],
        )
        self.assertTrue(first.osc_control_enabled)
        self.assertFalse(second.osc_control_enabled)

        server = Mock()
        server.start.return_value = ("127.0.0.4", 7101)
        with (
            patch(
                "transcriber.OscControlServer",
                return_value=server,
            ) as server_type,
            redirect_stdout(StringIO()),
        ):
            first.start_osc_control_server()
            second.start_osc_control_server()

        server_type.assert_called_once_with(
            "127.0.0.4",
            7101,
            first.apply_control,
            first.osc_logger.warning,
        )

    def test_vad_and_prompt_budgeting_use_the_pipeline_configuration(self):
        config = replace(
            RuntimeConfig(),
            vad_engine="energy",
            vad_energy_threshold=275.0,
            prompt_tokenizer_models=("tokenizer-a", "tokenizer-b"),
            prompt_max_tokens=61,
            prompt_min_transcript_tokens=17,
        )
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.side_effect = [Mock(), Mock()]
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = auto_tokenizer

        with (
            patch.dict(sys.modules, {"transformers": transformers}),
            redirect_stdout(StringIO()),
        ):
            pipeline = RealTimePipeline(
                enable_vad=True,
                enable_osc=False,
                enable_prompt_budget=True,
                enable_osc_controls=False,
                config=config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        self.assertEqual(pipeline.vad.threshold, 275.0)
        self.assertEqual(
            [call.args[0] for call in auto_tokenizer.from_pretrained.call_args_list],
            ["tokenizer-a", "tokenizer-b"],
        )
        self.assertEqual(pipeline.prompt_budgeter.max_tokens, 61)
        self.assertEqual(
            pipeline.prompt_budgeter.min_transcript_tokens,
            17,
        )

    def test_transcription_retries_use_the_pipeline_configuration(self):
        config = replace(
            RuntimeConfig(),
            transcription_final_max_retries=4,
            transcription_retry_base_seconds=2.25,
            transcription_retry_max_seconds=7.5,
        )
        pipeline = make_pipeline(config)
        pipeline.scheduler = Mock()
        pipeline.scheduler.retry_final.return_value = True
        pipeline.send_runtime_status = Mock()
        job = SimpleNamespace(
            is_final=True,
            attempts=1,
            segment=SimpleNamespace(segment_id=9),
        )

        with (
            patch("transcriber.time.monotonic", return_value=20.0),
            patch(
                "transcriber.exponential_backoff",
                return_value=4.5,
            ) as backoff,
            redirect_stdout(StringIO()),
        ):
            pipeline.handle_retryable_failure(
                job,
                RetryableTranscriptionError("temporary failure"),
            )

        backoff.assert_called_once_with(
            1,
            base_seconds=2.25,
            max_seconds=7.5,
        )
        pipeline.scheduler.retry_final.assert_called_once_with(
            job,
            20.0,
            4.5,
        )
        self.assertEqual(pipeline.backend_retry_not_before, 24.5)


class MicrophoneRecoveryTests(unittest.TestCase):
    def make_pipeline(self, config=None):
        pipeline = make_pipeline(config)
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
            pipeline.request_shutdown()
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
        pipeline = self.make_pipeline(
            replace(
                RuntimeConfig(),
                audio_max_consecutive_read_errors=2,
            )
        )
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

        with redirect_stdout(StringIO()):
            pipeline.audio_callback()

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(pipeline.audio_status, "stopped")
        pipeline.process_audio_data.assert_called_once()
        pipeline.finalize_interrupted_audio.assert_called_once()
        failing_stream.close.assert_called_once()
        recovered_stream.close.assert_called_once()

    def test_missing_pyaudio_has_an_actionable_runtime_error(self):
        pipeline = self.make_pipeline()

        with patch("transcriber.pyaudio", None):
            with self.assertRaisesRegex(RuntimeError, "--diagnose"):
                pipeline.create_audio_interface()

    def test_opens_the_selected_input_device(self):
        pipeline = self.make_pipeline(
            replace(RuntimeConfig(), audio_input_device_index=4)
        )
        audio_interface = Mock()
        selected = SimpleNamespace(index=4, name="Stage USB Microphone")

        with patch(
            "transcriber.get_audio_input_device",
            return_value=selected,
        ) as resolve_device:
            pipeline.open_microphone_stream(audio_interface)

        resolve_device.assert_called_once_with(audio_interface, 4)
        self.assertEqual(pipeline.audio_device_index, 4)
        self.assertEqual(pipeline.audio_device_name, "Stage USB Microphone")
        self.assertEqual(
            audio_interface.open.call_args.kwargs["input_device_index"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
