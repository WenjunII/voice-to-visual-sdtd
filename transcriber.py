import sys
import numpy as np
import pyaudio
import whisper
from orchestrator import PromptOrchestrator
import os
import torch

# --- Configuration ---
MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500 
SILENCE_DURATION = 1.5 

def main():
    try:
        print(f"Loading standard Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        print("Model loaded successfully.")
        
        print("Initializing Orchestrator...")
        orchestrator = PromptOrchestrator()
        print("Orchestrator ready.")
        
        p = pyaudio.PyAudio()
        print(f"Opening audio stream (Rate: {RATE})...")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("\n>>> Listening... Speak now (Ctrl+C to stop)")
        
        audio_buffer = []
        silent_chunks = 0
        
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            if np.abs(audio_data).mean() > SILENCE_THRESHOLD:
                audio_buffer.append(audio_data)
                silent_chunks = 0
            else:
                silent_chunks += 1
                
            if len(audio_buffer) > 0 and (silent_chunks * CHUNK / RATE) > SILENCE_DURATION:
                print("Processing speech chunk...")
                
                # Concatenate audio and normalize
                full_audio = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                
                # Transcribe using standard whisper
                # fp16=False if on CPU
                result = model.transcribe(full_audio, fp16=(DEVICE == "cuda"))
                
                text = result["text"].strip()
                
                if text:
                    orchestrator.refine_and_send(text)
                
                audio_buffer = []
                silent_chunks = 0
                
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()

if __name__ == "__main__":
    main()
