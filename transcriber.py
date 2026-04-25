import sys
import numpy as np
import pyaudio
import whisper
import torch
import threading
import time
from pythonosc import udp_client

# --- Configuration ---
MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OSC_IP = "127.0.0.1"
OSC_PORT = 7000

# --- FIXED PROMPT STRATEGY ---
FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text}, 8k UHD, highly detailed, masterfully lit, sharp focus, professional photography, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 400  # Lowered for better sensitivity
SILENCE_TIMEOUT = 5.0    # Increased to 5 seconds before resetting

class RealTimePipeline:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        
        self.audio_buffer = []
        self.last_text = ""
        self.is_running = True
        self.last_speech_time = time.time()
        
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
                        # Keep buffer at max 20 seconds
                        if len(self.audio_buffer) > (20 * RATE / CHUNK):
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
                if len(self.audio_buffer) < (RATE / CHUNK): # Need at least 1s of audio
                    continue
                full_audio = np.concatenate(self.audio_buffer).astype(np.float32) / 32768.0
            
            # Use no_speech_threshold to help Whisper ignore noise
            result = self.model.transcribe(full_audio, fp16=(DEVICE == "cuda"), language='en')
            text = result["text"].strip()
            
            # Filter out common Whisper hallucinations
            hallucinations = ["Thanks for watching", "Thank you", "Subtitle", "Subscribe"]
            if any(h.lower() in text.lower() for h in hallucinations):
                continue

            if text and text != self.last_text:
                self.last_text = text
                final_prompt = FIXED_PROMPT_TEMPLATE.format(text=text)
                
                self.osc_client.send_message("/prompt", final_prompt)
                self.osc_client.send_message("/partial_text", text)
                
                sys.stdout.write(f"\r[PROMPT]: {text[:80]}...         ")
                sys.stdout.flush()

    def start(self):
        t1 = threading.Thread(target=self.audio_callback, daemon=True)
        t2 = threading.Thread(target=self.transcription_loop, daemon=True)
        
        t1.start()
        t2.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running = False
            print("\nShutting down...")

if __name__ == "__main__":
    pipeline = RealTimePipeline()
    pipeline.start()
