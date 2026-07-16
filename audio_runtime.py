import numpy as np


class EnergyVoiceActivityDetector:
    def __init__(self, threshold):
        self.threshold = threshold

    def is_speech(self, audio_samples):
        return float(np.abs(audio_samples.astype(np.float32)).mean()) > self.threshold


class SileroVoiceActivityDetector:
    FRAME_SAMPLES = 512

    def __init__(self, threshold, sample_rate=16000):
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError("Silero VAD is not installed") from exc

        self.torch = torch
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = load_silero_vad(onnx=False)
        self.model.to("cpu")
        self.model.eval()

    def is_speech(self, audio_samples):
        normalized = audio_samples.astype(np.float32) / 32768.0
        probabilities = []
        with self.torch.inference_mode():
            for start in range(0, len(normalized), self.FRAME_SAMPLES):
                frame = normalized[start:start + self.FRAME_SAMPLES]
                if len(frame) < self.FRAME_SAMPLES:
                    frame = np.pad(frame, (0, self.FRAME_SAMPLES - len(frame)))
                tensor = self.torch.from_numpy(frame).to("cpu")
                probabilities.append(float(self.model(tensor, self.sample_rate).item()))
        return bool(probabilities) and max(probabilities) >= self.threshold
