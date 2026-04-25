import sys
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from orchestrator import PromptOrchestrator

# --- Configuration ---
MODEL_SIZE = "medium" # as requested
DEVICE = "cuda" # or "cpu"
COMPUTE_TYPE = "float16" # or "int8" for CPU

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500 # Adjust based on mic sensitivity
SILENCE_DURATION = 1.5 # Seconds of silence to trigger processing

def main():
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    orchestrator = PromptOrchestrator()
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n>>> Listening... Speak now (Ctrl+C to stop)")
    
    audio_buffer = []
    silent_chunks = 0
    
    try:
        while True:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Basic voice activity detection
            if np.abs(audio_data).mean() > SILENCE_THRESHOLD:
                audio_buffer.append(audio_data)
                silent_chunks = 0
            else:
                silent_chunks += 1
                
            # If we have audio and reached silence duration, process it
            if len(audio_buffer) > 0 and (silent_chunks * CHUNK / RATE) > SILENCE_DURATION:
                print("Processing speech chunk...")
                
                # Concatenate audio and normalize
                full_audio = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                
                # Transcribe
                segments, info = model.transcribe(full_audio, beam_size=5)
                
                text = " ".join([segment.text for segment in segments]).strip()
                
                if text:
                    orchestrator.refine_and_send(text)
                
                # Clear buffer
                audio_buffer = []
                silent_chunks = 0
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
