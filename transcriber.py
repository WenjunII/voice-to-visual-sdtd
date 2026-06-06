import os
import sys
import io
import wave
import argparse
import numpy as np
import pyaudio
import requests
import threading
import time
from pythonosc import udp_client
import msvcrt

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


def parse_args():
    parser = argparse.ArgumentParser(description="Live speech-to-visual prompt bridge for StreamDiffusionTD.")
    parser.add_argument(
        "-b",
        "--backend",
        choices=["whisper", "groq", "groq_hybrid", "google"],
        help="Transcription backend. Overrides TRANSCRIPTION_BACKEND from .env for this run."
    )
    return parser.parse_args()


load_env_file()
ARGS = parse_args()

# --- Configuration ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")
TRANSCRIPTION_BACKEND = (ARGS.backend or os.environ.get("TRANSCRIPTION_BACKEND", "whisper")).strip().lower()
if TRANSCRIPTION_BACKEND not in {"whisper", "groq", "groq_hybrid", "google"}:
    print(f"Unknown TRANSCRIPTION_BACKEND '{TRANSCRIPTION_BACKEND}', falling back to 'whisper'.")
    TRANSCRIPTION_BACKEND = "whisper"

torch = None
TORCH_IMPORT_ERROR = None
if TRANSCRIPTION_BACKEND == "whisper":
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

OSC_IP = "127.0.0.1"
OSC_PORT = 7000

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
    },
    "black_brown": {
        "label": "BLACK AND BROWN PEOPLE",
        "subject_prefix": "Black or Brown",
        "context": "centering Black and Brown people, contemporary US urban life, rich diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "contemporary US neighborhoods shaped by Black and Brown diasporic culture, warm natural color palettes, rich textures, vibrant lived-in atmosphere",
    },
    "asian_black_brown": {
        "label": "ASIAN + BLACK AND BROWN PEOPLE",
        "subject_prefix": "Asian, Black, or Brown",
        "context": "centering Asian, Black, and Brown people together, diverse contemporary US community life, layered diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "diverse contemporary US community spaces shaped by Asian, Black, and Brown diasporic culture, layered cultural textures, vibrant lived-in atmosphere",
    },
}

FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age_desc} {subject_focus}, {visual_context}, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"
SCENE_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic scene of {text}, {visual_context}, environment-focused composition, no central human figure, no portrait framing, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"

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

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 400  # Lowered for better sensitivity
SILENCE_TIMEOUT = 5.0    # Increased to 5 seconds before resetting

class RealTimePipeline:
    def __init__(self):
        self.backend = TRANSCRIPTION_BACKEND
        self.model = None
        self.online_recognizer = None
        self.last_online_request_time = 0
        self.last_whisper_request_time = 0
        self.http = requests.Session() if self.backend in {"groq", "groq_hybrid"} else None
        self.argos_package = None
        self.argos_translate = None
        self.local_translation_cache = {}

        if self.backend == "whisper":
            if torch is None:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=whisper requires torch. Run: pip install -r requirements.txt"
                ) from TORCH_IMPORT_ERROR
            if not DEVICE.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=whisper is configured to require a CUDA GPU. "
                    "Use TRANSCRIPTION_BACKEND=groq for online translation without local GPU usage."
                )
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=whisper requires openai-whisper. "
                    "Run: pip install -r requirements.txt"
                ) from exc

            print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
            self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)
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

        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        
        self.audio_buffer = []
        self.audio_version = 0
        self.last_submitted_audio_version = 0
        self.last_text = ""
        self.is_running = True
        self.last_speech_time = time.time()
        self.current_gender = CURRENT_GENDER
        self.current_age = CURRENT_AGE
        self.current_visual_mode = CURRENT_VISUAL_MODE
        self.current_prompt_style = CURRENT_PROMPT_STYLE
        self.current_language = None # Default to Auto
        
        self.lock = threading.Lock()
        
    def audio_callback(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("\n>>> Active. Visuals will update when you speak.")
        
        while self.is_running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                avg_vol = np.abs(audio_data).mean()
                
                with self.lock:
                    if avg_vol > SILENCE_THRESHOLD:
                        self.audio_buffer.append(audio_data)
                        self.audio_version += 1
                        self.last_speech_time = time.time()
                        # Keep buffer at max 12 seconds for SDXL token safety
                        if len(self.audio_buffer) > (12 * RATE / CHUNK):
                            self.audio_buffer.pop(0)
                    else:
                        # If silent for too long, clear the buffer to stop hallucinations
                        if time.time() - self.last_speech_time > SILENCE_TIMEOUT:
                            if self.audio_buffer:
                                self.audio_buffer = []
                                self.last_text = ""
            except Exception as e:
                print(f"Audio Error: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def transcription_loop(self):
        while self.is_running:
            time.sleep(0.6) 
            
            with self.lock:
                min_audio_seconds = self.minimum_audio_seconds()
                if len(self.audio_buffer) < (min_audio_seconds * RATE / CHUNK):
                    continue
                audio_version = self.audio_version
                if audio_version == self.last_submitted_audio_version:
                    continue
                full_audio = np.concatenate(self.audio_buffer).copy()

            if self.backend in {"groq", "groq_hybrid", "google"}:
                now = time.time()
                if now - self.last_online_request_time < self.online_request_interval():
                    continue
                self.last_online_request_time = now
                full_audio = self.limit_online_audio_window(full_audio)
            elif self.backend == "whisper":
                now = time.time()
                if now - self.last_whisper_request_time < WHISPER_TRANSCRIPTION_INTERVAL:
                    continue
                self.last_whisper_request_time = now
                full_audio = self.limit_whisper_audio_window(full_audio)

            self.last_submitted_audio_version = audio_version
            
            text = self.transcribe_audio(full_audio)
            
            # Filter out common recognizer hallucinations
            hallucinations = ["Thanks for watching", "Thank you", "Subtitle", "Subscribe"]
            if any(h.lower() in text.lower() for h in hallucinations):
                continue

            if text and text != self.last_text:
                self.last_text = text
                final_prompt = self.build_visual_prompt(text)
                
                self.osc_client.send_message("/prompt", final_prompt)
                self.osc_client.send_message("/partial_text", text)
                
                sys.stdout.write(f"\r[PROMPT]: {text[:80]}...         ")
                sys.stdout.flush()

    def build_visual_prompt(self, text):
        visual_mode = VISUAL_MODES.get(self.current_visual_mode, VISUAL_MODES[CURRENT_VISUAL_MODE])
        prompt_style = PROMPT_STYLES.get(self.current_prompt_style, PROMPT_STYLES[CURRENT_PROMPT_STYLE])

        if self.current_prompt_style == "general_scene":
            return prompt_style["template"].format(
                text=text,
                visual_context=visual_mode["scene_context"],
            )

        gender_focus = GENDER_MODES.get(self.current_gender, "person")
        age_desc = AGE_MODES.get(self.current_age, "")
        subject_focus = f"{visual_mode['subject_prefix']} {gender_focus}"

        return prompt_style["template"].format(
            age_desc=age_desc,
            subject_focus=subject_focus,
            visual_context=visual_mode["context"],
            text=text,
        )

    def transcribe_audio(self, audio_samples):
        if self.backend == "groq":
            return self.transcribe_groq(audio_samples)
        if self.backend == "groq_hybrid":
            return self.transcribe_groq_hybrid(audio_samples)
        if self.backend == "google":
            return self.transcribe_google(audio_samples)
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

    def limit_online_audio_window(self, audio_samples):
        max_seconds = GROQ_MAX_AUDIO_SECONDS if self.backend in {"groq", "groq_hybrid"} else GOOGLE_MAX_AUDIO_SECONDS
        max_samples = int(max_seconds * RATE)
        if max_samples <= 0 or len(audio_samples) <= max_samples:
            return audio_samples
        return audio_samples[-max_samples:]

    def limit_whisper_audio_window(self, audio_samples):
        max_samples = int(WHISPER_MAX_AUDIO_SECONDS * RATE)
        if max_samples <= 0 or len(audio_samples) <= max_samples:
            return audio_samples
        return audio_samples[-max_samples:]

    def transcribe_whisper(self, audio_samples):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        started = time.perf_counter()
        result = self.model.transcribe(
            full_audio,
            fp16=DEVICE.startswith("cuda"),
            task="translate",
            language=self.current_language,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        )
        elapsed = time.perf_counter() - started
        if WHISPER_LOG_LATENCY:
            print(f"\n[WHISPER LATENCY]: {elapsed:.2f}s for {len(audio_samples) / RATE:.1f}s audio")

        # Reverse segments so the latest speech has more weight at the start of the SDXL prompt.
        segments = result.get("segments", [])
        if segments:
            return " ".join([s["text"].strip() for s in reversed(segments)])
        return result["text"].strip()

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
            if response.status_code == 429:
                print("\n[GROQ TRANSCRIPTION RATE LIMIT]: Slow down or wait for the free quota to reset.")
                return ""
            if response.status_code >= 400:
                if response.status_code == 400 and "does not support `translate`" in response.text:
                    print(
                        "\n[GROQ TRANSCRIPTION ERROR]: Groq audio translation requires "
                        "GROQ_TRANSCRIPTION_MODEL=whisper-large-v3."
                    )
                    return ""
                print(f"\n[GROQ TRANSCRIPTION ERROR]: HTTP {response.status_code} {response.text[:160]}")
                return ""
            if GROQ_RESPONSE_FORMAT == "text":
                text = response.text.strip()
            else:
                text = response.json().get("text", "").strip()
            return self.ensure_english_prompt_text(text)
        except requests.RequestException as e:
            print(f"\n[GROQ TRANSCRIPTION ERROR]: {e}")
            return ""

    def transcribe_groq_hybrid(self, audio_samples):
        wav_bytes = self.encode_wav(audio_samples)
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {
            "model": GROQ_HYBRID_MODEL,
            "response_format": GROQ_RESPONSE_FORMAT,
            "temperature": "0",
        }
        if self.current_language:
            data["language"] = self.current_language

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
            if response.status_code == 429:
                print("\n[GROQ HYBRID RATE LIMIT]: Slow down or wait for the free quota to reset.")
                return ""
            if response.status_code >= 400:
                print(f"\n[GROQ HYBRID ERROR]: HTTP {response.status_code} {response.text[:160]}")
                return ""
            if GROQ_RESPONSE_FORMAT == "text":
                text = response.text.strip()
            else:
                text = response.json().get("text", "").strip()
            return self.ensure_local_english_prompt_text(text)
        except requests.RequestException as e:
            print(f"\n[GROQ HYBRID ERROR]: {e}")
            return ""

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
            if response.status_code >= 400:
                print(f"\n[GROQ TEXT TRANSLATION ERROR]: HTTP {response.status_code} {response.text[:160]}")
                return ""
            translation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if GROQ_LOG_LATENCY:
                print(f"\n[GROQ TEXT TRANSLATION]: {elapsed:.2f}s")
            return translation
        except requests.RequestException as e:
            print(f"\n[GROQ TEXT TRANSLATION ERROR]: {e}")
            return ""

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
        if self.current_language:
            return self.normalize_language_code(self.current_language)
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
            print(f"\n[ONLINE TRANSCRIPTION ERROR]: {e}")
            return ""

    def google_language(self):
        if self.current_language:
            return GOOGLE_LANGUAGE_MAP.get(self.current_language, self.current_language)
        return GOOGLE_SPEECH_LANGUAGE

    def start(self):
        t1 = threading.Thread(target=self.audio_callback, daemon=True)
        t2 = threading.Thread(target=self.transcription_loop, daemon=True)
        
        t1.start()
        t2.start()
        
        try:
            print("\n" + "="*50)
            print("CONTROL KEYS:")
            print("  [GENDER] 'm' -> Man | 'w' -> Woman | 'n' -> Neutral")
            print("  [AGE]    '1' -> Young | '2' -> Adult | '3' -> Elder")
            print("  [VISUAL] 'd' -> Asian American | 'b' -> Black and Brown people | 'x' -> Asian + Black and Brown")
            print("  [PROMPT] 'f' -> Human figure focus | 'g' -> General scene")
            print("  [LANG]   'e' -> English | 'c' -> Chinese | 's' -> Spanish | 'a' -> Auto")
            if self.backend == "groq":
                print("  [ONLINE] Groq translates detected speech to English automatically")
            if self.backend == "google":
                print(f"  [ONLINE] Auto/default language -> {GOOGLE_SPEECH_LANGUAGE}")
            print("  Ctrl+C   -> Exit")
            print("="*50 + "\n")
            
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 'm':
                        self.current_gender = "man"
                        print(f"\n[MODE]: GENDER -> MAN")
                    elif key == 'w':
                        self.current_gender = "woman"
                        print(f"\n[MODE]: GENDER -> WOMAN")
                    elif key == 'n':
                        self.current_gender = "neutral"
                        print(f"\n[MODE]: GENDER -> NEUTRAL")
                    elif key == '1':
                        self.current_age = "young"
                        print(f"\n[MODE]: AGE -> YOUNG")
                    elif key == '2':
                        self.current_age = "adult"
                        print(f"\n[MODE]: AGE -> ADULT")
                    elif key == '3':
                        self.current_age = "elder"
                        print(f"\n[MODE]: AGE -> ELDER")
                    elif key == 'd':
                        self.current_visual_mode = "asian_american"
                        print(f"\n[MODE]: VISUAL -> {VISUAL_MODES[self.current_visual_mode]['label']}")
                    elif key == 'b':
                        self.current_visual_mode = "black_brown"
                        print(f"\n[MODE]: VISUAL -> {VISUAL_MODES[self.current_visual_mode]['label']}")
                    elif key == 'x':
                        self.current_visual_mode = "asian_black_brown"
                        print(f"\n[MODE]: VISUAL -> {VISUAL_MODES[self.current_visual_mode]['label']}")
                    elif key == 'f':
                        self.current_prompt_style = "human_focus"
                        print(f"\n[MODE]: PROMPT -> {PROMPT_STYLES[self.current_prompt_style]['label']}")
                    elif key == 'g':
                        self.current_prompt_style = "general_scene"
                        print(f"\n[MODE]: PROMPT -> {PROMPT_STYLES[self.current_prompt_style]['label']}")
                    elif key == 'e':
                        self.current_language = "en"
                        print(f"\n[MODE]: LANG -> ENGLISH")
                    elif key == 'c':
                        self.current_language = "zh"
                        print(f"\n[MODE]: LANG -> CHINESE")
                    elif key == 's':
                        self.current_language = "es"
                        print(f"\n[MODE]: LANG -> SPANISH")
                    elif key == 'a':
                        self.current_language = None
                        print(f"\n[MODE]: LANG -> AUTO-DETECT")
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.is_running = False
            print("\nShutting down...")

if __name__ == "__main__":
    pipeline = RealTimePipeline()
    pipeline.start()
