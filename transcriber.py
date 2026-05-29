import os
import sys
import io
import wave
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


load_env_file()

# --- Configuration ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")
TRANSCRIPTION_BACKEND = os.environ.get("TRANSCRIPTION_BACKEND", "whisper").strip().lower()
if TRANSCRIPTION_BACKEND not in {"whisper", "groq", "google"}:
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

OSC_IP = "127.0.0.1"
OSC_PORT = 7000

# Online transcription is useful when TouchDesigner/StreamDiffusion needs the GPU.
# Groq's hosted Whisper translation endpoint keeps prompts in English without local GPU work.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")
GROQ_TRANSLATIONS_ENDPOINT = os.environ.get(
    "GROQ_TRANSLATIONS_ENDPOINT",
    "https://api.groq.com/openai/v1/audio/translations"
)
GROQ_TRANSCRIPTION_INTERVAL = env_float("GROQ_TRANSCRIPTION_INTERVAL", env_float("ONLINE_TRANSCRIPTION_INTERVAL", 5.0))
GROQ_MIN_AUDIO_SECONDS = env_float("GROQ_MIN_AUDIO_SECONDS", env_float("ONLINE_MIN_AUDIO_SECONDS", 2.0))
GROQ_MAX_AUDIO_SECONDS = env_float("GROQ_MAX_AUDIO_SECONDS", 5.0)
GROQ_REQUEST_TIMEOUT = env_float("GROQ_REQUEST_TIMEOUT", 20.0)

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
    "man": "Chinese-American man",
    "woman": "Chinese-American woman",
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

FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age_desc} {gender_focus}, capturing a diverse Chinese-American identity, blending modern US urban settings with subtle traditional Chinese cultural motifs and textures, 8k UHD, highly detailed, masterfully lit, fusion of East and West aesthetics, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"

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
        elif self.backend == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=groq requires GROQ_API_KEY in your environment or .env file."
                )
            print(f"Using Groq online Whisper translation backend ({GROQ_MODEL}). No local Whisper model loaded.")
            print("Note: this sends microphone audio to Groq and returns English text for StreamDiffusion prompts.")
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
        self.last_text = ""
        self.is_running = True
        self.last_speech_time = time.time()
        self.current_gender = CURRENT_GENDER
        self.current_age = CURRENT_AGE
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
                full_audio = np.concatenate(self.audio_buffer).copy()

            if self.backend in {"groq", "google"}:
                now = time.time()
                if now - self.last_online_request_time < self.online_request_interval():
                    continue
                self.last_online_request_time = now
                full_audio = self.limit_online_audio_window(full_audio)
            
            text = self.transcribe_audio(full_audio)
            
            # Filter out common recognizer hallucinations
            hallucinations = ["Thanks for watching", "Thank you", "Subtitle", "Subscribe"]
            if any(h.lower() in text.lower() for h in hallucinations):
                continue

            if text and text != self.last_text:
                self.last_text = text
                gender_focus = GENDER_MODES.get(self.current_gender, "person")
                age_desc = AGE_MODES.get(self.current_age, "")
                final_prompt = FIXED_PROMPT_TEMPLATE.format(age_desc=age_desc, gender_focus=gender_focus, text=text)
                
                self.osc_client.send_message("/prompt", final_prompt)
                self.osc_client.send_message("/partial_text", text)
                
                sys.stdout.write(f"\r[PROMPT]: {text[:80]}...         ")
                sys.stdout.flush()

    def transcribe_audio(self, audio_samples):
        if self.backend == "groq":
            return self.transcribe_groq(audio_samples)
        if self.backend == "google":
            return self.transcribe_google(audio_samples)
        return self.transcribe_whisper(audio_samples)

    def minimum_audio_seconds(self):
        if self.backend == "groq":
            return GROQ_MIN_AUDIO_SECONDS
        if self.backend == "google":
            return GOOGLE_MIN_AUDIO_SECONDS
        return 1.0

    def online_request_interval(self):
        if self.backend == "groq":
            return GROQ_TRANSCRIPTION_INTERVAL
        return GOOGLE_TRANSCRIPTION_INTERVAL

    def limit_online_audio_window(self, audio_samples):
        max_seconds = GROQ_MAX_AUDIO_SECONDS if self.backend == "groq" else GOOGLE_MAX_AUDIO_SECONDS
        max_samples = int(max_seconds * RATE)
        if max_samples <= 0 or len(audio_samples) <= max_samples:
            return audio_samples
        return audio_samples[-max_samples:]

    def transcribe_whisper(self, audio_samples):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        result = self.model.transcribe(
            full_audio,
            fp16=DEVICE.startswith("cuda"),
            task="translate",
            language=self.current_language
        )

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
            "response_format": "json",
            "temperature": "0",
        }

        try:
            response = requests.post(
                GROQ_TRANSLATIONS_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=GROQ_REQUEST_TIMEOUT
            )
            if response.status_code == 429:
                print("\n[GROQ TRANSCRIPTION RATE LIMIT]: Slow down or wait for the free quota to reset.")
                return ""
            if response.status_code >= 400:
                print(f"\n[GROQ TRANSCRIPTION ERROR]: HTTP {response.status_code} {response.text[:160]}")
                return ""
            return response.json().get("text", "").strip()
        except requests.RequestException as e:
            print(f"\n[GROQ TRANSCRIPTION ERROR]: {e}")
            return ""

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
