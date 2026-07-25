import os
import io
import wave
import argparse
import numpy as np
import requests
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
from backend_errors import RetryableTranscriptionError, exponential_backoff, retry_after_seconds
from diagnostics import run_diagnostics
from osc_control import OscControlServer
from prompt_engine import PromptBudgeter, RollingSceneMemory
from runtime_scheduler import RealtimeJobScheduler
from streaming_core import AudioSegmenter, TranscriptStabilizer
from transcript_filter import is_probable_whisper_hallucination

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from langdetect import detect as detect_language
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    detect_language = None
    LangDetectException = Exception


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


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


def env_optional_non_negative_int(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return optional_non_negative_int(value)
    except argparse.ArgumentTypeError:
        print(f"Invalid {name}='{value}'; using the system default input device.")
        return None


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
    return parser.parse_args(args)


load_env_file()
ARGS = parse_args() if __name__ == "__main__" else parse_args([])

# --- Configuration ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
TRANSCRIPTION_BACKEND = (ARGS.backend or os.environ.get("TRANSCRIPTION_BACKEND", "whisper")).strip().lower()
if TRANSCRIPTION_BACKEND not in {"whisper", "faster_whisper", "groq", "groq_hybrid", "google"}:
    print(f"Unknown TRANSCRIPTION_BACKEND '{TRANSCRIPTION_BACKEND}', falling back to 'whisper'.")
    TRANSCRIPTION_BACKEND = "whisper"

torch = None
TORCH_IMPORT_ERROR = None
if TRANSCRIPTION_BACKEND in {"whisper", "faster_whisper"}:
    try:
        import torch as torch_module
        torch = torch_module
    except Exception as exc:
        TORCH_IMPORT_ERROR = exc

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda").strip().lower()
if WHISPER_DEVICE:
    DEVICE = WHISPER_DEVICE
else:
    DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"

WHISPER_TRANSCRIPTION_INTERVAL = env_float("WHISPER_TRANSCRIPTION_INTERVAL", 0.8)
WHISPER_MIN_AUDIO_SECONDS = env_float("WHISPER_MIN_AUDIO_SECONDS", 0.8)
WHISPER_MAX_AUDIO_SECONDS = env_float("WHISPER_MAX_AUDIO_SECONDS", 6.0)
WHISPER_BEAM_SIZE = int(env_float("WHISPER_BEAM_SIZE", 1))
WHISPER_BEST_OF = int(env_float("WHISPER_BEST_OF", 1))
WHISPER_TEMPERATURE = env_float("WHISPER_TEMPERATURE", 0.0)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
WHISPER_LOG_LATENCY = env_bool("WHISPER_LOG_LATENCY", True)
FASTER_WHISPER_COMPUTE_TYPE = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8_float16").strip()
FASTER_WHISPER_CPU_THREADS = env_int("FASTER_WHISPER_CPU_THREADS", 4)
FASTER_WHISPER_NUM_WORKERS = env_int("FASTER_WHISPER_NUM_WORKERS", 1)

SCENE_MEMORY_MAX_WORDS = max(1, env_int("SCENE_MEMORY_MAX_WORDS", 36))
SCENE_MEMORY_MAX_AGE_SECONDS = env_float("SCENE_MEMORY_MAX_AGE_SECONDS", 20.0)
PROMPT_TOKEN_BUDGET_ENABLED = env_bool("PROMPT_TOKEN_BUDGET_ENABLED", True)
PROMPT_MAX_TOKENS = max(1, env_int("PROMPT_MAX_TOKENS", 77))
PROMPT_MIN_TRANSCRIPT_TOKENS = max(0, env_int("PROMPT_MIN_TRANSCRIPT_TOKENS", 20))
PROMPT_LOG_TOKENS = env_bool("PROMPT_LOG_TOKENS", True)
PROMPT_TOKENIZER_MODELS = [
    model.strip()
    for model in os.environ.get(
        "PROMPT_TOKENIZER_MODELS",
        "openai/clip-vit-large-patch14,laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ).split(",")
    if model.strip()
]

OSC_IP = os.environ.get("OSC_IP", "127.0.0.1")
OSC_PORT = env_int("OSC_PORT", 7000)
OSC_CONTROL_ENABLED = env_bool("OSC_CONTROL_ENABLED", True)
OSC_CONTROL_IP = os.environ.get("OSC_CONTROL_IP", "127.0.0.1")
OSC_CONTROL_PORT = env_int("OSC_CONTROL_PORT", 7001)
OSC_STATUS_INTERVAL = max(0.1, env_float("OSC_STATUS_INTERVAL", 0.5))

TRANSCRIPTION_MAX_FINAL_JOBS = max(1, env_int("TRANSCRIPTION_MAX_FINAL_JOBS", 8))
TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS = max(
    0.0,
    env_float("TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS", 4.0),
)
TRANSCRIPTION_FINAL_MAX_AGE_SECONDS = max(
    0.0,
    env_float("TRANSCRIPTION_FINAL_MAX_AGE_SECONDS", 30.0),
)
TRANSCRIPTION_FINAL_MAX_RETRIES = max(0, env_int("TRANSCRIPTION_FINAL_MAX_RETRIES", 2))
TRANSCRIPTION_RETRY_BASE_SECONDS = max(0.1, env_float("TRANSCRIPTION_RETRY_BASE_SECONDS", 1.0))
TRANSCRIPTION_RETRY_MAX_SECONDS = max(
    TRANSCRIPTION_RETRY_BASE_SECONDS,
    env_float("TRANSCRIPTION_RETRY_MAX_SECONDS", 10.0),
)

# Online transcription is useful when TouchDesigner/StreamDiffusion needs the GPU.
# Groq's hosted Whisper translation endpoint keeps prompts in English without local GPU work.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")
GROQ_HYBRID_MODEL = os.environ.get("GROQ_HYBRID_MODEL", "whisper-large-v3-turbo")
GROQ_TEXT_TRANSLATION_MODEL = os.environ.get("GROQ_TEXT_TRANSLATION_MODEL", "llama-3.1-8b-instant")
GROQ_TRANSCRIPTIONS_ENDPOINT = os.environ.get(
    "GROQ_TRANSCRIPTIONS_ENDPOINT",
    "https://api.groq.com/openai/v1/audio/transcriptions"
)
GROQ_TRANSLATIONS_ENDPOINT = os.environ.get(
    "GROQ_TRANSLATIONS_ENDPOINT",
    "https://api.groq.com/openai/v1/audio/translations"
)
GROQ_CHAT_ENDPOINT = os.environ.get(
    "GROQ_CHAT_ENDPOINT",
    "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_TRANSLATION_PROMPT = os.environ.get(
    "GROQ_TRANSLATION_PROMPT",
    "Translate all speech to natural concise English for a visual generation prompt."
)
GROQ_RESPONSE_FORMAT = os.environ.get("GROQ_RESPONSE_FORMAT", "text")
GROQ_TRANSCRIPTION_INTERVAL = env_float("GROQ_TRANSCRIPTION_INTERVAL", env_float("ONLINE_TRANSCRIPTION_INTERVAL", 3.2))
GROQ_MIN_AUDIO_SECONDS = env_float("GROQ_MIN_AUDIO_SECONDS", env_float("ONLINE_MIN_AUDIO_SECONDS", 1.0))
GROQ_MAX_AUDIO_SECONDS = env_float("GROQ_MAX_AUDIO_SECONDS", 6.0)
GROQ_REQUEST_TIMEOUT = env_float("GROQ_REQUEST_TIMEOUT", 20.0)
GROQ_LOG_LATENCY = env_bool("GROQ_LOG_LATENCY", True)
GROQ_ENGLISH_FALLBACK = os.environ.get("GROQ_ENGLISH_FALLBACK", "auto").strip().lower()

LOCAL_TRANSLATOR = os.environ.get("LOCAL_TRANSLATOR", "argos").strip().lower()
LOCAL_TRANSLATOR_TARGET_LANGUAGE = os.environ.get("LOCAL_TRANSLATOR_TARGET_LANGUAGE", "en").strip().lower()
LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE = os.environ.get("LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE", "zh").strip().lower()
LOCAL_TRANSLATOR_AUTO_INSTALL = env_bool("LOCAL_TRANSLATOR_AUTO_INSTALL", True)
LOCAL_TRANSLATOR_PRELOAD_LANGUAGES = [
    lang.strip().lower()
    for lang in os.environ.get("LOCAL_TRANSLATOR_PRELOAD_LANGUAGES", "zh,es").split(",")
    if lang.strip()
]
LOCAL_TRANSLATOR_LOG_LATENCY = env_bool("LOCAL_TRANSLATOR_LOG_LATENCY", True)
HYBRID_TRANSLATION_FALLBACK = os.environ.get("HYBRID_TRANSLATION_FALLBACK", "groq_text").strip().lower()

# This free Google path is recognition-only and does not translate to English.
GOOGLE_TRANSCRIPTION_INTERVAL = env_float("GOOGLE_TRANSCRIPTION_INTERVAL", env_float("ONLINE_TRANSCRIPTION_INTERVAL", 2.0))
GOOGLE_MIN_AUDIO_SECONDS = env_float("GOOGLE_MIN_AUDIO_SECONDS", env_float("ONLINE_MIN_AUDIO_SECONDS", 1.5))
GOOGLE_MAX_AUDIO_SECONDS = env_float("GOOGLE_MAX_AUDIO_SECONDS", 5.0)
GOOGLE_SPEECH_LANGUAGE = os.environ.get("GOOGLE_SPEECH_LANGUAGE", "en-US")
GOOGLE_LANGUAGE_MAP = {
    "en": os.environ.get("GOOGLE_SPEECH_ENGLISH_LANGUAGE", "en-US"),
    "zh": os.environ.get("GOOGLE_SPEECH_CHINESE_LANGUAGE", "zh-CN"),
    "es": os.environ.get("GOOGLE_SPEECH_SPANISH_LANGUAGE", "es-ES"),
}

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
AUDIO_INPUT_DEVICE_INDEX = (
    ARGS.input_device
    if ARGS.input_device is not None
    else env_optional_non_negative_int("AUDIO_INPUT_DEVICE_INDEX")
)
VAD_ENGINE = os.environ.get("VAD_ENGINE", "silero").strip().lower()
VAD_THRESHOLD = env_float("VAD_THRESHOLD", 0.5)
VAD_ENERGY_THRESHOLD = env_float("VAD_ENERGY_THRESHOLD", 400.0)
VAD_PRE_ROLL_SECONDS = env_float("VAD_PRE_ROLL_SECONDS", 0.32)
VAD_SILENCE_SECONDS = env_float("VAD_SILENCE_SECONDS", 0.7)
STREAM_OVERLAP_SECONDS = env_float("STREAM_OVERLAP_SECONDS", 0.5)
TRANSCRIPT_CONFIRM_UPDATES = max(1, env_int("TRANSCRIPT_CONFIRM_UPDATES", 2))
AUDIO_RECONNECT_ENABLED = env_bool("AUDIO_RECONNECT_ENABLED", True)
AUDIO_RECONNECT_BASE_SECONDS = max(
    0.1,
    env_float("AUDIO_RECONNECT_BASE_SECONDS", 0.5),
)
AUDIO_RECONNECT_MAX_SECONDS = max(
    AUDIO_RECONNECT_BASE_SECONDS,
    env_float("AUDIO_RECONNECT_MAX_SECONDS", 8.0),
)
AUDIO_MAX_CONSECUTIVE_READ_ERRORS = max(
    1,
    env_int("AUDIO_MAX_CONSECUTIVE_READ_ERRORS", 3),
)
AUDIO_READ_RETRY_SECONDS = max(
    0.0,
    env_float("AUDIO_READ_RETRY_SECONDS", 0.1),
)


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
    ):
        self.backend = TRANSCRIPTION_BACKEND
        self.model = None
        self.online_recognizer = None
        self.last_online_request_time = 0
        self.last_whisper_request_time = 0
        self.backend_retry_not_before = 0.0
        self.http = requests.Session() if self.backend in {"groq", "groq_hybrid"} else None
        self.argos_package = None
        self.argos_translate = None
        self.local_translation_cache = {}

        if self.backend in {"whisper", "faster_whisper"}:
            if torch is None:
                raise RuntimeError(
                    f"TRANSCRIPTION_BACKEND={self.backend} requires torch. Run: pip install -r requirements.txt"
                ) from TORCH_IMPORT_ERROR
            if not DEVICE.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError(
                    f"TRANSCRIPTION_BACKEND={self.backend} is configured to require a CUDA GPU. "
                    "Use TRANSCRIPTION_BACKEND=groq for online translation without local GPU usage."
                )

        if self.backend == "whisper":
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=whisper requires openai-whisper. "
                    "Run: pip install -r requirements.txt"
                ) from exc

            print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
            self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        elif self.backend == "faster_whisper":
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=faster_whisper requires faster-whisper. "
                    "Run: pip install -r requirements.txt"
                ) from exc

            print(
                f"Loading faster-whisper model '{MODEL_SIZE}' on {DEVICE} "
                f"({FASTER_WHISPER_COMPUTE_TYPE})..."
            )
            try:
                self.model = WhisperModel(
                    MODEL_SIZE,
                    device=DEVICE,
                    compute_type=FASTER_WHISPER_COMPUTE_TYPE,
                    cpu_threads=FASTER_WHISPER_CPU_THREADS,
                    num_workers=FASTER_WHISPER_NUM_WORKERS,
                )
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper could not initialize CUDA. This project pins CTranslate2 for "
                    "CUDA 12 with cuDNN 8; verify that the NVIDIA libraries are available on PATH."
                ) from exc
        elif self.backend in {"groq", "groq_hybrid"}:
            if not GROQ_API_KEY:
                raise RuntimeError(
                    f"TRANSCRIPTION_BACKEND={self.backend} requires GROQ_API_KEY in your environment or .env file."
                )
            if self.backend == "groq":
                print(f"Using Groq online Whisper translation backend ({GROQ_MODEL}). No local Whisper model loaded.")
                print("Note: this sends microphone audio to Groq and returns English text for StreamDiffusion prompts.")
            else:
                print(f"Using Groq hybrid backend ({GROQ_HYBRID_MODEL}) with local CPU text translation.")
                print("Note: Groq transcribes audio online; non-English text is translated locally when possible.")
                self.initialize_local_translator()
        else:
            if sr is None:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=google requires the SpeechRecognition package. "
                    "Run: pip install SpeechRecognition"
                )
            print("Using online Google Speech Recognition backend. No local Whisper model loaded.")
            print("Note: this sends microphone audio to Google's speech service and does not translate to English.")
            self.online_recognizer = sr.Recognizer()

        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT) if enable_osc else None
        self.osc_control_enabled = bool(enable_osc_controls and OSC_CONTROL_ENABLED)
        self.osc_control_server = None
        self.vad = self.create_voice_activity_detector() if enable_vad else None
        self.segmenter = AudioSegmenter(
            sample_rate=RATE,
            chunk_samples=CHUNK,
            pre_roll_seconds=VAD_PRE_ROLL_SECONDS,
            end_silence_seconds=VAD_SILENCE_SECONDS,
            max_segment_seconds=self.maximum_audio_seconds(),
            overlap_seconds=max(0.0, min(STREAM_OVERLAP_SECONDS, self.maximum_audio_seconds() / 2)),
        )
        self.scheduler = RealtimeJobScheduler(
            max_final_jobs=TRANSCRIPTION_MAX_FINAL_JOBS,
            partial_max_age_seconds=TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS,
            final_max_age_seconds=TRANSCRIPTION_FINAL_MAX_AGE_SECONDS,
        )
        self.stabilizers = {}
        self.scene_memory = RollingSceneMemory(
            max_words=SCENE_MEMORY_MAX_WORDS,
            max_age_seconds=SCENE_MEMORY_MAX_AGE_SECONDS,
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
        self.audio_device_index = AUDIO_INPUT_DEVICE_INDEX if AUDIO_INPUT_DEVICE_INDEX is not None else -1
        self.audio_device_name = "system default" if AUDIO_INPUT_DEVICE_INDEX is None else ""
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
        if not PROMPT_TOKEN_BUDGET_ENABLED:
            print("Prompt token budgeting is disabled.")
            return None
        try:
            from transformers import AutoTokenizer

            tokenizers = [AutoTokenizer.from_pretrained(model) for model in PROMPT_TOKENIZER_MODELS]
            print(
                f"Using {len(tokenizers)} SDXL CLIP tokenizer(s) "
                f"with a {PROMPT_MAX_TOKENS}-token prompt limit."
            )
            return PromptBudgeter(
                tokenizers,
                max_tokens=PROMPT_MAX_TOKENS,
                min_transcript_tokens=PROMPT_MIN_TRANSCRIPT_TOKENS,
            )
        except Exception as exc:
            print(f"[PROMPT BUDGET]: Tokenizers unavailable ({exc}). Prompts will not be token-limited.")
            return None

    def create_voice_activity_detector(self):
        if VAD_ENGINE == "silero":
            try:
                detector = SileroVoiceActivityDetector(VAD_THRESHOLD, sample_rate=RATE)
                print(f"Using Silero VAD on CPU (threshold {VAD_THRESHOLD:.2f}).")
                return detector
            except Exception as exc:
                print(f"[VAD]: Silero unavailable ({exc}). Falling back to energy detection.")
        elif VAD_ENGINE != "energy":
            print(f"[VAD]: Unknown engine '{VAD_ENGINE}'. Falling back to energy detection.")

        print(f"Using energy VAD (threshold {VAD_ENERGY_THRESHOLD:.0f}).")
        return EnergyVoiceActivityDetector(VAD_ENERGY_THRESHOLD)

    def create_audio_interface(self):
        if pyaudio is None:
            raise RuntimeError(
                "PyAudio is not installed. Run 'python transcriber.py --diagnose' "
                "for setup details, then install the project requirements."
            )
        return pyaudio.PyAudio()

    def open_microphone_stream(self, audio_interface):
        device = get_audio_input_device(audio_interface, AUDIO_INPUT_DEVICE_INDEX)
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
            self.vad = EnergyVoiceActivityDetector(VAD_ENERGY_THRESHOLD)
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

                        if consecutive_read_errors >= AUDIO_MAX_CONSECUTIVE_READ_ERRORS:
                            raise RuntimeError(
                                f"microphone read failed {consecutive_read_errors} consecutive times: {exc}"
                            ) from exc

                        print(
                            f"\n[AUDIO READ ERROR]: {exc} "
                            f"({consecutive_read_errors}/{AUDIO_MAX_CONSECUTIVE_READ_ERRORS}); "
                            "retrying the current stream."
                        )
                        if not self.wait_for_audio_retry(AUDIO_READ_RETRY_SECONDS):
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
            if not AUDIO_RECONNECT_ENABLED:
                self.audio_status = "error"
                self.is_running = False
                print(f"\n[AUDIO ERROR]: {failure}. Automatic reconnection is disabled.")
                self.send_runtime_status(force=True)
                break

            retry_delay = exponential_backoff(
                reconnect_attempt,
                base_seconds=AUDIO_RECONNECT_BASE_SECONDS,
                max_seconds=AUDIO_RECONNECT_MAX_SECONDS,
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
                TranscriptStabilizer(TRANSCRIPT_CONFIRM_UPDATES),
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
        if self.backend in {"groq", "groq_hybrid", "google"}:
            return now - self.last_online_request_time >= self.online_request_interval()
        return now - self.last_whisper_request_time >= self.local_request_interval()

    def mark_request_started(self, now):
        if self.backend in {"groq", "groq_hybrid", "google"}:
            self.last_online_request_time = now
        else:
            self.last_whisper_request_time = now

    def handle_retryable_failure(self, job, exc):
        now = time.monotonic()
        retry_delay = exc.retry_after
        if retry_delay is None:
            retry_delay = exponential_backoff(
                job.attempts,
                base_seconds=TRANSCRIPTION_RETRY_BASE_SECONDS,
                max_seconds=TRANSCRIPTION_RETRY_MAX_SECONDS,
            )
        self.backend_retry_not_before = max(
            self.backend_retry_not_before,
            now + retry_delay,
        )

        should_retry = job.is_final and job.attempts < TRANSCRIPTION_FINAL_MAX_RETRIES
        if should_retry and self.scheduler.retry_final(job, now, retry_delay):
            self.backend_status = "retrying"
            print(
                f"\n[TRANSCRIPTION RETRY]: Final segment {job.segment.segment_id} "
                f"in {retry_delay:.1f}s ({job.attempts + 1}/{TRANSCRIPTION_FINAL_MAX_RETRIES})."
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
            if not force and now - self.last_status_osc_time < OSC_STATUS_INTERVAL:
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
        if PROMPT_LOG_TOKENS:
            trimmed = ", newest transcript retained" if result.transcript_trimmed else ""
            print(
                f"\n[PROMPT TOKENS]: {result.token_count}/{PROMPT_MAX_TOKENS} "
                f"({result.variant}{trimmed})"
            )
        return result.text

    def transcribe_audio(self, audio_samples):
        if self.backend == "groq":
            return self.transcribe_groq(audio_samples)
        if self.backend == "groq_hybrid":
            return self.transcribe_groq_hybrid(audio_samples)
        if self.backend == "google":
            return self.transcribe_google(audio_samples)
        if self.backend == "faster_whisper":
            return self.transcribe_faster_whisper(audio_samples)
        return self.transcribe_whisper(audio_samples)

    def minimum_audio_seconds(self):
        if self.backend in {"groq", "groq_hybrid"}:
            return GROQ_MIN_AUDIO_SECONDS
        if self.backend == "google":
            return GOOGLE_MIN_AUDIO_SECONDS
        return WHISPER_MIN_AUDIO_SECONDS

    def online_request_interval(self):
        if self.backend in {"groq", "groq_hybrid"}:
            return GROQ_TRANSCRIPTION_INTERVAL
        return GOOGLE_TRANSCRIPTION_INTERVAL

    def local_request_interval(self):
        if self.backend in {"whisper", "faster_whisper"}:
            return WHISPER_TRANSCRIPTION_INTERVAL
        return 0.0

    def maximum_audio_seconds(self):
        if self.backend in {"groq", "groq_hybrid"}:
            return max(1.0, GROQ_MAX_AUDIO_SECONDS)
        if self.backend == "google":
            return max(1.0, GOOGLE_MAX_AUDIO_SECONDS)
        return max(1.0, WHISPER_MAX_AUDIO_SECONDS)

    def transcribe_whisper(self, audio_samples):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        started = time.perf_counter()
        result = self.model.transcribe(
            full_audio,
            fp16=DEVICE.startswith("cuda"),
            task="translate",
            language=self.selected_language(),
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        )
        elapsed = time.perf_counter() - started
        if WHISPER_LOG_LATENCY:
            print(f"\n[WHISPER LATENCY]: {elapsed:.2f}s for {len(audio_samples) / RATE:.1f}s audio")

        segments = result.get("segments", [])
        if segments:
            return " ".join(s["text"].strip() for s in segments if s["text"].strip())
        return result["text"].strip()

    def transcribe_faster_whisper(self, audio_samples):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        started = time.perf_counter()
        segments, _ = self.model.transcribe(
            full_audio,
            task="translate",
            language=self.selected_language(),
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            vad_filter=False,
            word_timestamps=False,
        )
        segments = list(segments)
        elapsed = time.perf_counter() - started
        if WHISPER_LOG_LATENCY:
            print(f"\n[FASTER-WHISPER LATENCY]: {elapsed:.2f}s for {len(audio_samples) / RATE:.1f}s audio")
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())

    @staticmethod
    def raise_for_retryable_http_response(response, operation):
        if response.status_code == 429:
            raise RetryableTranscriptionError(
                f"{operation} was rate-limited by Groq (HTTP 429)",
                retry_after=retry_after_seconds(response.headers),
            )
        if response.status_code in {408, 409, 425} or response.status_code >= 500:
            raise RetryableTranscriptionError(
                f"{operation} is temporarily unavailable (HTTP {response.status_code})",
                retry_after=retry_after_seconds(response.headers),
            )

    def transcribe_groq(self, audio_samples):
        wav_bytes = self.encode_wav(audio_samples)
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {
            "model": GROQ_MODEL,
            "response_format": GROQ_RESPONSE_FORMAT,
            "temperature": "0",
        }
        if GROQ_TRANSLATION_PROMPT:
            data["prompt"] = GROQ_TRANSLATION_PROMPT

        try:
            started = time.perf_counter()
            response = self.http.post(
                GROQ_TRANSLATIONS_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=GROQ_REQUEST_TIMEOUT
            )
            elapsed = time.perf_counter() - started
            if GROQ_LOG_LATENCY:
                print(f"\n[GROQ LATENCY]: {elapsed:.2f}s for {len(audio_samples) / RATE:.1f}s audio")
            self.raise_for_retryable_http_response(response, "Groq transcription")
            if response.status_code >= 400:
                if response.status_code == 400 and "does not support `translate`" in response.text:
                    raise RuntimeError(
                        "Groq audio translation requires "
                        "GROQ_TRANSCRIPTION_MODEL=whisper-large-v3"
                    )
                raise RuntimeError(
                    f"Groq transcription rejected the request (HTTP {response.status_code}): "
                    f"{response.text[:160]}"
                )
            if GROQ_RESPONSE_FORMAT == "text":
                text = response.text.strip()
            else:
                text = response.json().get("text", "").strip()
            return self.ensure_english_prompt_text(text)
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq transcription request failed: {exc}"
            ) from exc

    def transcribe_groq_hybrid(self, audio_samples):
        wav_bytes = self.encode_wav(audio_samples)
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {
            "model": GROQ_HYBRID_MODEL,
            "response_format": GROQ_RESPONSE_FORMAT,
            "temperature": "0",
        }
        language = self.selected_language()
        if language:
            data["language"] = language

        try:
            started = time.perf_counter()
            response = self.http.post(
                GROQ_TRANSCRIPTIONS_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=GROQ_REQUEST_TIMEOUT
            )
            elapsed = time.perf_counter() - started
            if GROQ_LOG_LATENCY:
                print(f"\n[GROQ HYBRID LATENCY]: {elapsed:.2f}s for {len(audio_samples) / RATE:.1f}s audio")
            self.raise_for_retryable_http_response(response, "Groq hybrid transcription")
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Groq hybrid transcription rejected the request (HTTP {response.status_code}): "
                    f"{response.text[:160]}"
                )
            if GROQ_RESPONSE_FORMAT == "text":
                text = response.text.strip()
            else:
                text = response.json().get("text", "").strip()
            return self.ensure_local_english_prompt_text(text)
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq hybrid transcription request failed: {exc}"
            ) from exc

    def ensure_english_prompt_text(self, text):
        if not text:
            return ""
        if GROQ_ENGLISH_FALLBACK == "off":
            return text
        if GROQ_ENGLISH_FALLBACK == "always" or self.contains_cjk(text):
            translated = self.translate_text_to_english(text)
            return translated or text
        return text

    def contains_cjk(self, text):
        return any(
            "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"
            for char in text
        )

    def translate_text_to_english(self, text):
        payload = {
            "model": GROQ_TEXT_TRANSLATION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. Translate the user's text completely into natural concise "
                        "English for a visual generation prompt. Output only English text. Do not include any "
                        "Chinese characters."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 120,
        }

        try:
            started = time.perf_counter()
            response = self.http.post(
                GROQ_CHAT_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=GROQ_REQUEST_TIMEOUT,
            )
            elapsed = time.perf_counter() - started
            self.raise_for_retryable_http_response(response, "Groq text translation")
            if response.status_code >= 400:
                print(f"\n[GROQ TEXT TRANSLATION ERROR]: HTTP {response.status_code} {response.text[:160]}")
                return ""
            translation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if GROQ_LOG_LATENCY:
                print(f"\n[GROQ TEXT TRANSLATION]: {elapsed:.2f}s")
            return translation
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq text translation request failed: {exc}"
            ) from exc

    def initialize_local_translator(self):
        if LOCAL_TRANSLATOR != "argos":
            print(f"[LOCAL TRANSLATOR]: Unknown translator '{LOCAL_TRANSLATOR}'. Non-English text will pass through.")
            return
        try:
            import argostranslate.package as argos_package
            import argostranslate.translate as argos_translate
        except ImportError:
            print("[LOCAL TRANSLATOR]: Argos Translate is not installed. Run: pip install argostranslate langdetect")
            return
        except Exception as e:
            print(f"[LOCAL TRANSLATOR]: Argos Translate could not load: {e}")
            print("[LOCAL TRANSLATOR]: Non-English hybrid transcripts will pass through untranslated.")
            return

        self.argos_package = argos_package
        self.argos_translate = argos_translate
        for source_code in LOCAL_TRANSLATOR_PRELOAD_LANGUAGES:
            self.get_argos_translation(source_code)

    def ensure_local_english_prompt_text(self, text):
        if not text:
            return ""
        source_language = self.detect_text_language(text)
        if not source_language or source_language == LOCAL_TRANSLATOR_TARGET_LANGUAGE:
            return text
        translated = self.translate_text_locally(text, source_language)
        if translated and not self.contains_cjk(translated):
            return translated
        if HYBRID_TRANSLATION_FALLBACK == "groq_text":
            fallback = self.translate_text_to_english(text)
            if fallback:
                return fallback
        return translated or text

    def detect_text_language(self, text):
        language = self.selected_language()
        if language:
            return self.normalize_language_code(language)
        if self.contains_cjk(text):
            return "zh"
        if detect_language is not None and len(text.strip()) >= 8:
            try:
                return self.normalize_language_code(detect_language(text))
            except LangDetectException:
                pass
        if self.looks_like_english(text):
            return "en"
        return LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE

    def normalize_language_code(self, language_code):
        code = (language_code or "").lower()
        if code.startswith("zh"):
            return "zh"
        if code.startswith("es"):
            return "es"
        if code.startswith("en"):
            return "en"
        return code

    def looks_like_english(self, text):
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return True
        ascii_letters = [char for char in letters if "a" <= char.lower() <= "z"]
        return len(ascii_letters) / len(letters) > 0.85

    def translate_text_locally(self, text, source_language):
        if LOCAL_TRANSLATOR != "argos":
            return ""
        translation = self.get_argos_translation(source_language)
        if translation is None:
            return ""
        try:
            started = time.perf_counter()
            translated = translation.translate(text).strip()
            elapsed = time.perf_counter() - started
            if LOCAL_TRANSLATOR_LOG_LATENCY:
                print(f"\n[LOCAL TRANSLATION]: {source_language}->en {elapsed:.2f}s")
            return translated
        except Exception as e:
            print(f"\n[LOCAL TRANSLATION ERROR]: {e}")
            return ""

    def get_argos_translation(self, source_language):
        source_language = self.normalize_language_code(source_language)
        cache_key = (source_language, LOCAL_TRANSLATOR_TARGET_LANGUAGE)
        if cache_key in self.local_translation_cache:
            return self.local_translation_cache[cache_key]
        if self.argos_translate is None:
            return None

        translation = self.find_argos_translation(source_language)
        if translation is None and LOCAL_TRANSLATOR_AUTO_INSTALL:
            self.install_argos_package(source_language)
            translation = self.find_argos_translation(source_language)

        if translation is None:
            print(f"\n[LOCAL TRANSLATION]: No Argos package for {source_language}->en. Text will pass through.")
        self.local_translation_cache[cache_key] = translation
        return translation

    def find_argos_translation(self, source_language):
        installed_languages = self.argos_translate.get_installed_languages()
        from_language = next((lang for lang in installed_languages if lang.code == source_language), None)
        to_language = next((lang for lang in installed_languages if lang.code == LOCAL_TRANSLATOR_TARGET_LANGUAGE), None)
        if not from_language or not to_language:
            return None
        try:
            return from_language.get_translation(to_language)
        except Exception:
            return None

    def install_argos_package(self, source_language):
        if self.argos_package is None:
            return
        try:
            print(f"\n[LOCAL TRANSLATION]: Installing Argos package {source_language}->en...")
            self.argos_package.update_package_index()
            available_packages = self.argos_package.get_available_packages()
            package = next(
                (
                    pkg for pkg in available_packages
                    if pkg.from_code == source_language and pkg.to_code == LOCAL_TRANSLATOR_TARGET_LANGUAGE
                ),
                None
            )
            if package is None:
                print(f"\n[LOCAL TRANSLATION]: No downloadable Argos package for {source_language}->en.")
                return
            download_path = package.download()
            self.argos_package.install_from_path(download_path)
        except Exception as e:
            print(f"\n[LOCAL TRANSLATION ERROR]: Could not install {source_language}->en package: {e}")

    def encode_wav(self, audio_samples):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(RATE)
            wav_file.writeframes(audio_samples.astype(np.int16).tobytes())
        buffer.seek(0)
        return buffer.read()

    def transcribe_google(self, audio_samples):
        audio_data = sr.AudioData(audio_samples.astype(np.int16).tobytes(), RATE, 2)
        language = self.google_language()

        try:
            return self.online_recognizer.recognize_google(audio_data, language=language).strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise RetryableTranscriptionError(f"Google transcription request failed: {e}") from e

    def google_language(self):
        language = self.selected_language()
        if language:
            return GOOGLE_LANGUAGE_MAP.get(language, language)
        return GOOGLE_SPEECH_LANGUAGE

    def benchmark(self, wav_path, runs=3):
        if self.backend not in {"whisper", "faster_whisper"}:
            raise RuntimeError("Benchmark mode supports only whisper and faster_whisper backends.")

        samples = self.load_wav_file(wav_path)
        audio_seconds = len(samples) / RATE
        runs = max(1, runs)
        warmup = samples[:min(len(samples), int(2 * RATE))]

        print("\n" + "=" * 50)
        print(f"LOCAL ASR BENCHMARK: {self.backend}")
        print(f"Model: {MODEL_SIZE} | Device: {DEVICE} | Audio: {audio_seconds:.2f}s")
        if self.backend == "faster_whisper":
            print(f"Compute type: {FASTER_WHISPER_COMPUTE_TYPE}")
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
            OSC_CONTROL_IP,
            OSC_CONTROL_PORT,
            self.apply_control,
        )
        try:
            address = server.start()
        except OSError as exc:
            print(f"[OSC CONTROL]: Could not bind {OSC_CONTROL_IP}:{OSC_CONTROL_PORT} ({exc}).")
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
            print(f"BACKEND: {self.backend} | VAD: {VAD_ENGINE} | CONFIRMATIONS: {TRANSCRIPT_CONFIRM_UPDATES}")
            print("CONTROL KEYS:")
            print("  [GENDER] 'm' -> Man | 'w' -> Woman | 'n' -> Neutral")
            print("  [AGE]    '1' -> Young | '2' -> Adult | '3' -> Elder")
            print("  [VISUAL] 'd' -> Asian American | 'b' -> Black and Brown people | 'x' -> Asian + Black and Brown")
            print("  [PROMPT] 'f' -> Human figure focus | 'g' -> General scene")
            print("  [LANG]   'e' -> English | 'c' -> Chinese | 's' -> Spanish | 'a' -> Auto")
            if self.osc_control_server is not None:
                print(f"  [OSC IN] {OSC_CONTROL_IP}:{OSC_CONTROL_PORT} for TouchDesigner controls")
            if self.backend == "groq":
                print("  [ONLINE] Groq translates detected speech to English automatically")
            if self.backend == "google":
                print(f"  [ONLINE] Auto/default language -> {GOOGLE_SPEECH_LANGUAGE}")
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

if __name__ == "__main__":
    if ARGS.list_audio_devices:
        raise SystemExit(print_audio_input_devices())

    if ARGS.diagnose:
        raise SystemExit(
            run_diagnostics(
                backend=TRANSCRIPTION_BACKEND,
                model_size=MODEL_SIZE,
                device=DEVICE,
                sample_rate=RATE,
                chunk_size=CHUNK,
                osc_control_ip=OSC_CONTROL_IP,
                osc_control_port=OSC_CONTROL_PORT,
                groq_key_configured=bool(GROQ_API_KEY),
                capriole_key_configured=bool(os.environ.get("CAPRIOLE_API_KEY")),
                audio_input_device_index=AUDIO_INPUT_DEVICE_INDEX,
            )
        )

    pipeline = RealTimePipeline(
        enable_vad=not bool(ARGS.benchmark),
        enable_osc=not bool(ARGS.benchmark),
        enable_prompt_budget=not bool(ARGS.benchmark),
        enable_osc_controls=not bool(ARGS.benchmark),
    )
    if ARGS.benchmark:
        pipeline.benchmark(ARGS.benchmark, ARGS.benchmark_runs)
    else:
        pipeline.start()
