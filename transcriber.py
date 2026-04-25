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
# Refined for Photo-Realistic SDXL output
FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text}, 8k UHD, highly detailed, masterfully lit, sharp focus, professional photography, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 300 

class RealTimePipeline:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        
        self.audio_buffer = []
        self.last_text = ""
        self.is_running = True
        
        self.lock = threading.Lock()
        
    def audio_callback(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("\n>>> Fixed Prompt Pipeline Active. Visuals will update CONSTANTLY.")
        
        while self.is_running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                with self.lock:
                    self.audio_buffer.append(audio_data)
                    # Keep buffer at max 20 seconds for real-time responsiveness
                    if len(self.audio_buffer) > (20 * RATE / CHUNK):
                        self.audio_buffer.pop(0)
            except Exception as e:
                print(f"Audio Error: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def transcription_loop(self):
        while self.is_running:
            time.sleep(0.5) # Fast 500ms updates for instant visual changes
            
            with self.lock:
                if not self.audio_buffer:
                    continue
                full_audio = np.concatenate(self.audio_buffer).astype(np.float32) / 32768.0
            
            # 1. Transcribe the current buffer (fast)
            # We use a short window to keep the visuals moving
            result = self.model.transcribe(full_audio, fp16=(DEVICE == "cuda"), language='en')
            text = result["text"].strip()
            
            if text and text != self.last_text:
                self.last_text = text
                
                # 2. Apply the FIXED PROMPT TEMPLATE
                # This bypasses LLM latency for instant updates
                final_prompt = FIXED_PROMPT_TEMPLATE.format(text=text)
                
                # 3. Send to TouchDesigner
                self.osc_client.send_message("/prompt", final_prompt)
                self.osc_client.send_message("/partial_text", text)
                
                print(f"\r[PROMPT]: {final_prompt[:90]}...", end="")
                sys.stdout.flush()

            # Optional: Clear buffer if a very long silence is detected (e.g. 5 seconds)
            # This helps "reset" the visual theme when you stop talking.
            # ... (omitted for pure constant mode)

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
