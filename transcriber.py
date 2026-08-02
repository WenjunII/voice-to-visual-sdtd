import os
import wave
import argparse
import numpy as np
import threading
import time
from pythonosc import udp_client
import msvcrt

from audio_runtime import (
    EnergyVoiceActivityDetector,
    SileroVoiceActivityDetector,
    get_audio_input_device,
    list_audio_input_devices,
)
from backend_errors import RetryableTranscriptionError, exponential_backoff
from diagnostics import run_diagnostics
from osc_control import OscControlServer
from prompt_engine import PromptBudgeter, RollingSceneMemory
from runtime_config import (
    ConfigError,
    RuntimeConfig,
    format_config_error,
    format_config_report,
    load_env_file,
)
from runtime_scheduler import RealtimeJobScheduler
from streaming_core import AudioSegmenter, TranscriptStabilizer
from transcript_filter import is_probable_whisper_hallucination
from transcription_backends import create_transcription_backend

try:
    import pyaudio
except ImportError:
    pyaudio = None

def optional_non_negative_int(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Live speech-to-visual prompt bridge for StreamDiffusionTD.")
    parser.add_argument(
        "-b",
        "--backend",
        choices=["whisper", "faster_whisper", "groq", "groq_hybrid", "google"],
        help="Transcription backend. Overrides TRANSCRIPTION_BACKEND from .env for this run."
    )
    parser.add_argument(
        "--input-device",
        type=optional_non_negative_int,
        metavar="INDEX",
        help="PyAudio input-device index. Overrides AUDIO_INPUT_DEVICE_INDEX from .env."
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List input-capable audio devices and exit."
    )
    parser.add_argument(
        "--benchmark",
        metavar="WAV_PATH",
        help="Benchmark a local backend with a PCM WAV file instead of opening the microphone."
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of measured benchmark runs after warm-up (default: 3)."
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check the GPU, microphone, packages, OSC port, model cache, and credential hygiene."
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print the effective configuration with secrets redacted."
    )
    return parser.parse_args(args)


def load_runtime_config(args=None):
    load_env_file()
    return RuntimeConfig.from_environment(
        backend_override=getattr(args, "backend", None),
        input_device_override=getattr(args, "input_device", None),
    )


def configure_dependency_environment():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- FIXED PROMPT STRATEGY ---
# GENDER MODES: Press 'm' for Man, 'w' for Woman, 'n' for Neutral (General)
GENDER_MODES = {
    "man": "man",
    "woman": "woman",
    "neutral": "person"
}

# AGE MODES: Press '1' for Young, '2' for Adult, '3' for Elder
AGE_MODES = {
    "young": "young",
    "adult": "adult",
    "elder": "elderly"
}

CURRENT_GENDER = "neutral"
CURRENT_AGE = "adult"
CURRENT_VISUAL_MODE = "asian_american"
CURRENT_PROMPT_STYLE = "human_focus"

VISUAL_MODES = {
    "asian_american": {
        "label": "ASIAN AMERICAN",
        "subject_prefix": "Asian-American",
        "context": "capturing a diverse Asian-American identity, blending modern US urban settings with subtle traditional Asian cultural motifs and textures",
        "scene_context": "modern Asian-American neighborhoods and interiors, subtle traditional Asian cultural motifs, layered urban textures, natural cinematic atmosphere",
        "compact_context": "Asian-American US setting with subtle Asian cultural motifs",
        "compact_scene_context": "Asian-American US setting with subtle Asian cultural motifs and cinematic atmosphere",
    },
    "black_brown": {
        "label": "BLACK AND BROWN PEOPLE",
        "subject_prefix": "Black or Brown",
        "context": "centering Black and Brown people, contemporary US urban life, rich diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "contemporary US neighborhoods shaped by Black and Brown diasporic culture, warm natural color palettes, rich textures, vibrant lived-in atmosphere",
        "compact_context": "Black and Brown diasporic US setting with warm tones and rich textures",
        "compact_scene_context": "Black and Brown diasporic US setting with warm colors and rich textures",
    },
    "asian_black_brown": {
        "label": "ASIAN + BLACK AND BROWN PEOPLE",
        "subject_prefix": "Asian, Black, or Brown",
        "context": "centering Asian, Black, and Brown people together, diverse contemporary US community life, layered diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "diverse contemporary US community spaces shaped by Asian, Black, and Brown diasporic culture, layered cultural textures, vibrant lived-in atmosphere",
        "compact_context": "diverse Asian, Black, and Brown diasporic US community",
        "compact_scene_context": "diverse Asian, Black, and Brown diasporic US community spaces",
    },
}

FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age_desc} {subject_focus}, {visual_context}, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"
SCENE_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic scene of {text}, {visual_context}, environment-focused composition, no central human figure, no portrait framing, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"
COMPACT_FIXED_PROMPT_TEMPLATE = "Photorealistic cinematic scene: {text}, prominent {age_desc} {subject_focus}, {visual_context}, highly detailed, natural colors, masterfully lit, RAW 35mm photo, f/1.8, 8k"
COMPACT_SCENE_PROMPT_TEMPLATE = "Photorealistic cinematic scene: {text}, {visual_context}, environment-focused, no central human figure, highly detailed, natural colors, masterfully lit, RAW 35mm photo, f/1.8, 8k"

PROMPT_STYLES = {
    "human_focus": {
        "label": "HUMAN FIGURE",
        "template": FIXED_PROMPT_TEMPLATE,
    },
    "general_scene": {
        "label": "GENERAL SCENE",
        "template": SCENE_PROMPT_TEMPLATE,
    },
}

KEYBOARD_CONTROLS = {
    "m": ("gender", "man"),
    "w": ("gender", "woman"),
    "n": ("gender", "neutral"),
    "1": ("age", "young"),
    "2": ("age", "adult"),
    "3": ("age", "elder"),
    "d": ("visual_mode", "asian_american"),
    "b": ("visual_mode", "black_brown"),
    "x": ("visual_mode", "asian_black_brown"),
    "f": ("prompt_style", "human_focus"),
    "g": ("prompt_style", "general_scene"),
    "e": ("language", "en"),
    "c": ("language", "zh"),
    "s": ("language", "es"),
    "a": ("language", None),
}

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16 if pyaudio is not None else None
CHANNELS = 1
RATE = 16000


def print_audio_input_devices():
    if pyaudio is None:
        print("PyAudio is not installed; audio devices cannot be listed.")
        return 1

    audio_interface = None
    try:
        audio_interface = pyaudio.PyAudio()
        devices = list_audio_input_devices(audio_interface)
        if not devices:
            print("No input-capable audio devices were found.")
            return 1

        print("\nINPUT AUDIO DEVICES")
        print("=" * 72)
        for device in devices:
            marker = "*" if device.is_default else " "
            rate = f"{device.default_sample_rate:.0f} Hz" if device.default_sample_rate else "unknown rate"
            print(
                f"{marker} [{device.index}] {device.name} "
                f"({device.max_input_channels} input channel(s), {rate})"
            )
        print("=" * 72)
        print("* system default")
        return 0
    except Exception as exc:
        print(f"Could not enumerate audio devices: {exc}")
        return 1
    finally:
        if audio_interface is not None:
            audio_interface.terminate()


class RealTimePipeline:
    def __init__(
        self,
        enable_vad=True,
        enable_osc=True,
        enable_prompt_budget=True,
        enable_osc_controls=True,
        config=None,
        backend_adapter=None,
    ):
        configure_dependency_environment()
        self.config = (
            config if config is not None else load_runtime_config()
        )
        self.backend_adapter = (
            backend_adapter
            if backend_adapter is not None
            else create_transcription_backend(
                self.config,
                sample_rate=RATE,
            )
        )
        self._backend_closed = False
        self.backend = self.backend_adapter.name
        self.last_online_request_time = 0
        self.last_whisper_request_time = 0
        self.backend_retry_not_before = 0.0

        self.osc_client = (
            udp_client.SimpleUDPClient(
                self.config.osc_ip,
                self.config.osc_port,
            )
            if enable_osc
            else None
        )
        self.osc_control_enabled = bool(
            enable_osc_controls and self.config.osc_control_enabled
        )
        self.osc_control_server = None
        self.vad = self.create_voice_activity_detector() if enable_vad else None
        self.segmenter = AudioSegmenter(
            sample_rate=RATE,
            chunk_samples=CHUNK,
            pre_roll_seconds=self.config.vad_pre_roll_seconds,
            end_silence_seconds=self.config.vad_silence_seconds,
            max_segment_seconds=self.maximum_audio_seconds(),
            overlap_seconds=max(
                0.0,
                min(
                    self.config.stream_overlap_seconds,
                    self.maximum_audio_seconds() / 2,
                ),
            ),
        )
        self.scheduler = RealtimeJobScheduler(
            max_final_jobs=self.config.transcription_max_final_jobs,
            partial_max_age_seconds=(
                self.config.transcription_partial_max_age_seconds
            ),
            final_max_age_seconds=(
                self.config.transcription_final_max_age_seconds
            ),
        )
        self.stabilizers = {}
        self.scene_memory = RollingSceneMemory(
            max_words=self.config.scene_memory_max_words,
            max_age_seconds=self.config.scene_memory_max_age_seconds,
        )
        self.prompt_budgeter = self.create_prompt_budgeter() if enable_prompt_budget else None
        self.last_prompt_token_count = 0
        self.last_prompt_variant = "unbudgeted"
        self.last_prompt_trimmed = False
        self.last_text = ""
        self.is_running = True
        self.is_speaking = False
        self.audio_status = "starting"
        self.audio_reconnects = 0
        self.last_audio_error = ""
        configured_device = self.config.audio_input_device_index
        self.audio_device_index = (
            configured_device if configured_device is not None else -1
        )
        self.audio_device_name = (
            "system default" if configured_device is None else ""
        )
        self.backend_status = "ready"
        self.last_inference_latency = 0.0
        self.last_total_latency = 0.0
        self.last_status_osc_time = 0.0
        self.current_gender = CURRENT_GENDER
        self.current_age = CURRENT_AGE
        self.current_visual_mode = CURRENT_VISUAL_MODE
        self.current_prompt_style = CURRENT_PROMPT_STYLE
        self.current_language = None  # Default to Auto
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.scene_lock = threading.Lock()
        self.osc_lock = threading.Lock()

    def create_prompt_budgeter(self):
        if not self.config.prompt_token_budget_enabled:
            print("Prompt token budgeting is disabled.")
            return None
        try:
            from transformers import AutoTokenizer

            tokenizers = [
                AutoTokenizer.from_pretrained(model)
                for model in self.config.prompt_tokenizer_models
            ]
            print(
                f"Using {len(tokenizers)} SDXL CLIP tokenizer(s) "
                f"with a {self.config.prompt_max_tokens}-token prompt limit."
            )
            return PromptBudgeter(
                tokenizers,
                max_tokens=self.config.prompt_max_tokens,
                min_transcript_tokens=(
                    self.config.prompt_min_transcript_tokens
                ),
            )
        except Exception as exc:
            print(f"[PROMPT BUDGET]: Tokenizers unavailable ({exc}). Prompts will not be token-limited.")
            return None

    def create_voice_activity_detector(self):
        if self.config.vad_engine == "silero":
            try:
                detector = SileroVoiceActivityDetector(
                    self.config.vad_threshold,
                    sample_rate=RATE,
                )
                print(
                    "Using Silero VAD on CPU "
                    f"(threshold {self.config.vad_threshold:.2f})."
                )
                return detector
            except Exception as exc:
                print(f"[VAD]: Silero unavailable ({exc}). Falling back to energy detection.")
        elif self.config.vad_engine != "energy":
            print(
                f"[VAD]: Unknown engine '{self.config.vad_engine}'. "
                "Falling back to energy detection."
            )

        print(
            "Using energy VAD "
            f"(threshold {self.config.vad_energy_threshold:.0f})."
        )
        return EnergyVoiceActivityDetector(
            self.config.vad_energy_threshold
        )

    def create_audio_interface(self):
        if pyaudio is None:
            raise RuntimeError(
                "PyAudio is not installed. Run 'python transcriber.py --diagnose' "
                "for setup details, then install the project requirements."
            )
        return pyaudio.PyAudio()

    def open_microphone_stream(self, audio_interface):
        device = get_audio_input_device(
            audio_interface,
            self.config.audio_input_device_index,
        )
        self.audio_device_index = device.index
        self.audio_device_name = device.name
        return audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=CHUNK,
        )

    def process_audio_data(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16)
        try:
            is_speech = self.vad.is_speech(audio_data)
        except Exception as exc:
            print(f"\n[VAD ERROR]: {exc}. Switching to energy detection.")
            self.vad = EnergyVoiceActivityDetector(
                self.config.vad_energy_threshold
            )
            is_speech = self.vad.is_speech(audio_data)

        with self.lock:
            completed = self.segmenter.add_chunk(audio_data, is_speech)
            speaking = self.segmenter.active

        if speaking != self.is_speaking:
            self.is_speaking = speaking
            self.send_runtime_status(force=True)

        now = time.monotonic()
        for segment in completed:
            if self.scheduler.submit_final(segment, now) is None:
                print("\n[SCHEDULER]: Final queue is full; newest final segment was dropped.")
        self.send_runtime_status()

    def wait_for_audio_retry(self, delay_seconds):
        deadline = time.monotonic() + max(0.0, delay_seconds)
        while self.is_running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.1, remaining))
        return False

    @staticmethod
    def close_audio_session(stream, audio_interface):
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
            except Exception:
                pass

    def finalize_interrupted_audio(self):
        with self.lock:
            segment = self.segmenter.interrupt()
        if segment is None or segment.samples.size == 0:
            return
        if self.scheduler.submit_final(segment, time.monotonic()) is None:
            print("\n[SCHEDULER]: Final queue is full; interrupted audio was dropped.")

    def audio_callback(self):
        reconnect_attempt = 0

        while self.is_running:
            audio_interface = None
            stream = None
            failure = None
            try:
                audio_interface = self.create_audio_interface()
                stream = self.open_microphone_stream(audio_interface)

                if self.audio_reconnects:
                    print(
                        f"\n[AUDIO RECOVERED]: [{self.audio_device_index}] "
                        f"{self.audio_device_name} is available again."
                    )
                else:
                    print(
                        f"\n>>> Active on [{self.audio_device_index}] {self.audio_device_name}. "
                        "Visuals will update when you speak."
                    )
                self.audio_status = "ready"
                self.last_audio_error = ""
                self.send_runtime_status(force=True)

                consecutive_read_errors = 0
                while self.is_running:
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                    except Exception as exc:
                        consecutive_read_errors += 1
                        self.audio_status = "degraded"
                        self.last_audio_error = self.clean_audio_error(exc)
                        self.is_speaking = False
                        self.send_runtime_status(force=True)

                        max_read_errors = (
                            self.config.audio_max_consecutive_read_errors
                        )
                        if consecutive_read_errors >= max_read_errors:
                            raise RuntimeError(
                                f"microphone read failed {consecutive_read_errors} consecutive times: {exc}"
                            ) from exc

                        print(
                            f"\n[AUDIO READ ERROR]: {exc} "
                            f"({consecutive_read_errors}/{max_read_errors}); "
                            "retrying the current stream."
                        )
                        if not self.wait_for_audio_retry(
                            self.config.audio_read_retry_seconds
                        ):
                            break
                        continue

                    if consecutive_read_errors:
                        print("\n[AUDIO RECOVERED]: Microphone reads resumed.")
                        self.audio_status = "ready"
                        self.last_audio_error = ""
                        self.send_runtime_status(force=True)
                    consecutive_read_errors = 0
                    reconnect_attempt = 0
                    try:
                        self.process_audio_data(data)
                    except Exception as exc:
                        print(f"\n[AUDIO PROCESSING ERROR]: {exc}")
            except Exception as exc:
                failure = exc
            finally:
                self.close_audio_session(stream, audio_interface)

            if failure is None or not self.is_running:
                break

            if stream is not None:
                self.finalize_interrupted_audio()
            self.is_speaking = False
            self.last_audio_error = self.clean_audio_error(failure)
            if not self.config.audio_reconnect_enabled:
                self.audio_status = "error"
                self.is_running = False
                print(f"\n[AUDIO ERROR]: {failure}. Automatic reconnection is disabled.")
                self.send_runtime_status(force=True)
                break

            retry_delay = exponential_backoff(
                reconnect_attempt,
                base_seconds=self.config.audio_reconnect_base_seconds,
                max_seconds=self.config.audio_reconnect_max_seconds,
            )
            reconnect_attempt += 1
            self.audio_reconnects += 1
            self.audio_status = "reconnecting"
            print(
                f"\n[AUDIO RECONNECT]: {failure}. "
                f"Retrying in {retry_delay:.1f}s (attempt {self.audio_reconnects})."
            )
            self.send_runtime_status(force=True)
            if not self.wait_for_audio_retry(retry_delay):
                break

        if self.audio_status != "error":
            self.audio_status = "stopped"
            self.is_speaking = False
            self.send_runtime_status(force=True)

    @staticmethod
    def clean_audio_error(error):
        return " ".join(str(error).split())[:240]

    def transcription_loop(self):
        while self.is_running:
            now = time.monotonic()
            if not self.request_interval_ready(now):
                self.send_runtime_status()
                time.sleep(0.05)
                continue

            with self.lock:
                partial = self.segmenter.snapshot(self.minimum_audio_seconds())
            if partial is not None:
                self.scheduler.submit_partial(partial, now)

            job = self.scheduler.next_job(now)
            if job is None:
                self.send_runtime_status()
                time.sleep(0.05)
                continue

            self.mark_request_started(now)
            self.backend_status = "transcribing"
            self.send_runtime_status(force=True)
            started = time.monotonic()
            try:
                text = self.transcribe_audio(job.segment.samples)
            except RetryableTranscriptionError as exc:
                self.handle_retryable_failure(job, exc)
                continue
            except Exception as exc:
                print(f"\n[TRANSCRIPTION ERROR]: {exc}")
                self.scheduler.mark_failed()
                self.backend_status = "error"
                self.cleanup_final_job(job)
                self.send_runtime_status(force=True)
                continue

            finished = time.monotonic()
            self.last_inference_latency = finished - started
            self.last_total_latency = finished - job.created_at
            self.scheduler.mark_processed()
            self.backend_status = "ready"
            self.backend_retry_not_before = 0.0

            if is_probable_whisper_hallucination(text):
                text = ""

            stabilizer = self.stabilizers.setdefault(
                job.segment.segment_id,
                TranscriptStabilizer(
                    self.config.transcript_confirm_updates
                ),
            )
            update = stabilizer.update(text, is_final=job.is_final)
            scene_text = ""
            if update.text:
                with self.scene_lock:
                    scene_text = self.scene_memory.update(
                        job.segment.segment_id,
                        update.text,
                        is_final=update.is_final,
                    )
            if scene_text and (update.changed or update.is_final):
                self.emit_transcript(
                    scene_text,
                    raw_text=update.text,
                    is_final=update.is_final,
                )

            self.cleanup_final_job(job)
            self.send_runtime_status(force=True)

    def request_interval_ready(self, now):
        if now < self.backend_retry_not_before:
            return False
        if self.backend_adapter.online:
            return now - self.last_online_request_time >= self.online_request_interval()
        return now - self.last_whisper_request_time >= self.local_request_interval()

    def mark_request_started(self, now):
        if self.backend_adapter.online:
            self.last_online_request_time = now
        else:
            self.last_whisper_request_time = now

    def handle_retryable_failure(self, job, exc):
        now = time.monotonic()
        retry_delay = exc.retry_after
        if retry_delay is None:
            retry_delay = exponential_backoff(
                job.attempts,
                base_seconds=(
                    self.config.transcription_retry_base_seconds
                ),
                max_seconds=(
                    self.config.transcription_retry_max_seconds
                ),
            )
        self.backend_retry_not_before = max(
            self.backend_retry_not_before,
            now + retry_delay,
        )

        max_retries = self.config.transcription_final_max_retries
        should_retry = job.is_final and job.attempts < max_retries
        if should_retry and self.scheduler.retry_final(job, now, retry_delay):
            self.backend_status = "retrying"
            print(
                f"\n[TRANSCRIPTION RETRY]: Final segment {job.segment.segment_id} "
                f"in {retry_delay:.1f}s "
                f"({job.attempts + 1}/{max_retries})."
            )
        else:
            self.scheduler.mark_failed()
            self.backend_status = "error"
            self.cleanup_final_job(job)
            print(f"\n[TRANSCRIPTION ERROR]: {exc}")
        self.send_runtime_status(force=True)

    def cleanup_final_job(self, job):
        if job.is_final:
            self.stabilizers.pop(job.segment.segment_id, None)

    def send_osc_message(self, address, value):
        if self.osc_client is None:
            return
        with self.osc_lock:
            self.osc_client.send_message(address, value)

    def send_runtime_status(self, force=False):
        if self.osc_client is None:
            return
        now = time.monotonic()
        with self.state_lock:
            current_gender = self.current_gender
            current_age = self.current_age
            current_visual_mode = self.current_visual_mode
            current_prompt_style = self.current_prompt_style
            current_language = self.current_language or "auto"
        with self.osc_lock:
            if (
                not force
                and now - self.last_status_osc_time
                < self.config.osc_status_interval
            ):
                return
            metrics = self.scheduler.metrics(now)
            self.osc_client.send_message("/backend_status", self.backend_status)
            self.osc_client.send_message("/backend", self.backend)
            self.osc_client.send_message("/is_speaking", int(self.is_speaking))
            self.osc_client.send_message("/queue_depth", metrics.queue_depth)
            self.osc_client.send_message("/latency_total", float(self.last_total_latency))
            self.osc_client.send_message("/latency_asr", float(self.last_inference_latency))
            self.osc_client.send_message(
                "/retry_in",
                float(max(0.0, self.backend_retry_not_before - now)),
            )
            self.osc_client.send_message(
                "/dropped_jobs",
                metrics.dropped_stale + metrics.dropped_finals,
            )
            self.osc_client.send_message("/audio_status", self.audio_status)
            self.osc_client.send_message("/audio_reconnects", self.audio_reconnects)
            self.osc_client.send_message("/audio_error", self.last_audio_error)
            self.osc_client.send_message("/audio_device_index", self.audio_device_index)
            self.osc_client.send_message("/audio_device_name", self.audio_device_name)
            self.osc_client.send_message("/gender", current_gender)
            self.osc_client.send_message("/age", current_age)
            self.osc_client.send_message("/visual_mode", current_visual_mode)
            self.osc_client.send_message("/prompt_style", current_prompt_style)
            self.osc_client.send_message("/language", current_language)
            self.last_status_osc_time = now

    def selected_language(self):
        with self.state_lock:
            return self.current_language

    def emit_transcript(self, text, raw_text=None, is_final=False):
        if not text:
            return
        raw_text = raw_text or text
        with self.scene_lock:
            unchanged = text == self.last_text
            if not unchanged:
                self.last_text = text
        if unchanged:
            if is_final:
                self.send_osc_message("/transcript_final", raw_text)
            return
        final_prompt = self.build_visual_prompt(text)

        self.send_osc_message("/prompt", final_prompt)
        self.send_osc_message("/partial_text", raw_text)
        self.send_osc_message("/scene_context", text)
        self.send_osc_message("/prompt_tokens", self.last_prompt_token_count)
        if is_final:
            self.send_osc_message("/transcript_final", raw_text)

        state = "FINAL" if is_final else "STABLE"
        print(f"\n[PROMPT {state}]: {text}")
        if raw_text != text:
            print(f"[CURRENT TRANSCRIPT]: {raw_text}")

    def refresh_visual_prompt(self):
        with self.scene_lock:
            text = self.last_text
        if not text:
            return False

        final_prompt = self.build_visual_prompt(text)
        self.send_osc_message("/prompt", final_prompt)
        self.send_osc_message("/prompt_tokens", self.last_prompt_token_count)
        print(f"\n[PROMPT REFRESH]: Applied the current visual controls to: {text}")
        return True

    def build_visual_prompt(self, text):
        with self.state_lock:
            current_visual_mode = self.current_visual_mode
            current_prompt_style = self.current_prompt_style
            current_gender = self.current_gender
            current_age = self.current_age

        visual_mode = VISUAL_MODES.get(current_visual_mode, VISUAL_MODES[CURRENT_VISUAL_MODE])

        if current_prompt_style == "general_scene":
            variants = [
                (
                    "full",
                    lambda value: SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["scene_context"],
                    ),
                ),
                (
                    "compact_context",
                    lambda value: SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["compact_scene_context"],
                    ),
                ),
                (
                    "compact",
                    lambda value: COMPACT_SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["compact_scene_context"],
                    ),
                ),
            ]
        else:
            gender_focus = GENDER_MODES.get(current_gender, "person")
            age_desc = AGE_MODES.get(current_age, "")
            subject_focus = f"{visual_mode['subject_prefix']} {gender_focus}"
            variants = [
                (
                    "full",
                    lambda value: FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["context"],
                        text=value,
                    ),
                ),
                (
                    "compact_context",
                    lambda value: FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["compact_context"],
                        text=value,
                    ),
                ),
                (
                    "compact",
                    lambda value: COMPACT_FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["compact_context"],
                        text=value,
                    ),
                ),
            ]

        if self.prompt_budgeter is None:
            self.last_prompt_token_count = 0
            self.last_prompt_variant = "unbudgeted"
            self.last_prompt_trimmed = False
            return variants[0][1](text)

        result = self.prompt_budgeter.fit(variants, text)
        self.last_prompt_token_count = result.token_count
        self.last_prompt_variant = result.variant
        self.last_prompt_trimmed = result.transcript_trimmed
        if self.config.prompt_log_tokens:
            trimmed = ", newest transcript retained" if result.transcript_trimmed else ""
            print(
                f"\n[PROMPT TOKENS]: {result.token_count}/"
                f"{self.config.prompt_max_tokens} "
                f"({result.variant}{trimmed})"
            )
        return result.text

    def transcribe_audio(self, audio_samples):
        return self.backend_adapter.transcribe(
            audio_samples,
            language=self.selected_language(),
        )

    def minimum_audio_seconds(self):
        return self.backend_adapter.minimum_audio_seconds

    def online_request_interval(self):
        if self.backend_adapter.online:
            return self.backend_adapter.request_interval
        return 0.0

    def local_request_interval(self):
        if not self.backend_adapter.online:
            return self.backend_adapter.request_interval
        return 0.0

    def maximum_audio_seconds(self):
        return self.backend_adapter.maximum_audio_seconds

    def benchmark(self, wav_path, runs=3):
        if self.backend not in {"whisper", "faster_whisper"}:
            raise RuntimeError("Benchmark mode supports only whisper and faster_whisper backends.")

        samples = self.load_wav_file(wav_path)
        audio_seconds = len(samples) / RATE
        runs = max(1, runs)
        warmup = samples[:min(len(samples), int(2 * RATE))]

        print("\n" + "=" * 50)
        print(f"LOCAL ASR BENCHMARK: {self.backend}")
        device = getattr(
            self.backend_adapter,
            "device",
            self.config.whisper_device or "auto",
        )
        print(
            f"Model: {self.config.whisper_model_size} | "
            f"Device: {device} | Audio: {audio_seconds:.2f}s"
        )
        if self.backend == "faster_whisper":
            print(
                "Compute type: "
                f"{self.config.faster_whisper_compute_type}"
            )
        print("Warming up...")
        self.transcribe_audio(warmup)

        latencies = []
        for run_number in range(1, runs + 1):
            started = time.perf_counter()
            text = self.transcribe_audio(samples)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            rtf = elapsed / audio_seconds
            print(f"Run {run_number}: {elapsed:.3f}s | real-time factor {rtf:.3f} | {len(text)} chars")

        average = sum(latencies) / len(latencies)
        print(f"Average: {average:.3f}s | real-time factor {average / audio_seconds:.3f}")
        print("=" * 50)

    @staticmethod
    def load_wav_file(wav_path):
        with wave.open(wav_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_audio = wav_file.readframes(frame_count)

        if sample_width != 2:
            raise ValueError("Benchmark WAV must use 16-bit PCM audio.")

        samples = np.frombuffer(raw_audio, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
        if source_rate != RATE:
            target_length = max(1, round(len(samples) * RATE / source_rate))
            source_positions = np.linspace(0, len(samples) - 1, num=len(samples))
            target_positions = np.linspace(0, len(samples) - 1, num=target_length)
            samples = np.interp(target_positions, source_positions, samples).astype(np.int16)
        if samples.size == 0:
            raise ValueError("Benchmark WAV contains no audio samples.")
        return samples

    def start_osc_control_server(self):
        if not self.osc_control_enabled:
            return
        server = OscControlServer(
            self.config.osc_control_ip,
            self.config.osc_control_port,
            self.apply_control,
        )
        try:
            address = server.start()
        except OSError as exc:
            print(
                "[OSC CONTROL]: Could not bind "
                f"{self.config.osc_control_ip}:"
                f"{self.config.osc_control_port} ({exc})."
            )
            return
        self.osc_control_server = server
        print(f"[OSC CONTROL]: Listening on {address[0]}:{address[1]}.")

    def apply_control(self, control_name, value):
        if control_name == "request_status":
            self.send_runtime_status(force=True)
            return
        if control_name == "reset_scene":
            with self.scene_lock:
                self.scene_memory.reset()
                self.last_text = ""
            self.send_osc_message("/scene_context", "")
            self.send_osc_message("/scene_reset", 1)
            print("\n[MODE]: SCENE MEMORY -> RESET")
            return

        valid_values = {
            "gender": GENDER_MODES,
            "age": AGE_MODES,
            "visual_mode": VISUAL_MODES,
            "prompt_style": PROMPT_STYLES,
            "language": {None: None, "en": "ENGLISH", "zh": "CHINESE", "es": "SPANISH"},
        }
        if control_name not in valid_values or value not in valid_values[control_name]:
            raise ValueError(f"Invalid {control_name} control value: {value}")

        with self.state_lock:
            if control_name == "gender":
                self.current_gender = value
                label = value.upper()
            elif control_name == "age":
                self.current_age = value
                label = value.upper()
            elif control_name == "visual_mode":
                self.current_visual_mode = value
                label = VISUAL_MODES[value]["label"]
            elif control_name == "prompt_style":
                self.current_prompt_style = value
                label = PROMPT_STYLES[value]["label"]
            else:
                self.current_language = value
                label = "AUTO-DETECT" if value is None else valid_values["language"][value]

        print(f"\n[MODE]: {control_name.upper()} -> {label}")
        acknowledgement = "auto" if value is None else str(value)
        self.send_osc_message("/control_ack", f"{control_name}:{acknowledgement}")
        if control_name in {"gender", "age", "visual_mode", "prompt_style"}:
            self.refresh_visual_prompt()
        self.send_runtime_status(force=True)

    def start(self):
        self.start_osc_control_server()
        t1 = threading.Thread(target=self.audio_callback, daemon=True)
        t2 = threading.Thread(target=self.transcription_loop, daemon=True)
        
        t1.start()
        t2.start()
        
        try:
            print("\n" + "="*50)
            print(
                f"BACKEND: {self.backend} | "
                f"VAD: {self.config.vad_engine} | "
                f"CONFIRMATIONS: {self.config.transcript_confirm_updates}"
            )
            print("CONTROL KEYS:")
            print("  [GENDER] 'm' -> Man | 'w' -> Woman | 'n' -> Neutral")
            print("  [AGE]    '1' -> Young | '2' -> Adult | '3' -> Elder")
            print("  [VISUAL] 'd' -> Asian American | 'b' -> Black and Brown people | 'x' -> Asian + Black and Brown")
            print("  [PROMPT] 'f' -> Human figure focus | 'g' -> General scene")
            print("  [LANG]   'e' -> English | 'c' -> Chinese | 's' -> Spanish | 'a' -> Auto")
            if self.osc_control_server is not None:
                print(
                    f"  [OSC IN] {self.config.osc_control_ip}:"
                    f"{self.config.osc_control_port} "
                    "for TouchDesigner controls"
                )
            if self.backend == "groq":
                print("  [ONLINE] Groq translates detected speech to English automatically")
            if self.backend == "google":
                print(
                    "  [ONLINE] Auto/default language -> "
                    f"{self.config.google_speech_language}"
                )
            print("  Ctrl+C   -> Exit")
            print("="*50 + "\n")
            self.send_runtime_status(force=True)
            
            while self.is_running:
                if msvcrt.kbhit():
                    try:
                        key = msvcrt.getch().decode("utf-8").lower()
                    except UnicodeDecodeError:
                        continue
                    control = KEYBOARD_CONTROLS.get(key)
                    if control is not None:
                        self.apply_control(*control)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.is_running = False
            self.backend_status = "stopped"
            self.is_speaking = False
            self.send_runtime_status(force=True)
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)
            if self.osc_control_server is not None:
                self.osc_control_server.stop()
            self.close()

    def close(self):
        if self._backend_closed:
            return
        self._backend_closed = True
        self.backend_adapter.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_runtime_config(args)
    except ConfigError as exc:
        print(format_config_error(exc))
        return 2

    if args.check_config:
        print(format_config_report(config))
        return 0

    if args.list_audio_devices:
        return print_audio_input_devices()

    if args.diagnose:
        return run_diagnostics(
            config=config,
            sample_rate=RATE,
            chunk_size=CHUNK,
            device=config.whisper_device or "cuda",
        )

    pipeline = RealTimePipeline(
        enable_vad=not bool(args.benchmark),
        enable_osc=not bool(args.benchmark),
        enable_prompt_budget=not bool(args.benchmark),
        enable_osc_controls=not bool(args.benchmark),
        config=config,
    )
    try:
        if args.benchmark:
            pipeline.benchmark(args.benchmark, args.benchmark_runs)
        else:
            pipeline.start()
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
