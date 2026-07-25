from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioInputDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False


def get_audio_input_device(audio_interface, index=None):
    """Resolve and validate a PyAudio input device."""

    if index is None:
        info = audio_interface.get_default_input_device_info()
    else:
        info = audio_interface.get_device_info_by_index(index)

    device_index = int(info.get("index", index if index is not None else -1))
    max_input_channels = int(info.get("maxInputChannels", 0))
    if max_input_channels < 1:
        raise ValueError(f"audio device {device_index} has no input channels")

    return AudioInputDevice(
        index=device_index,
        name=str(info.get("name", f"device {device_index}")),
        max_input_channels=max_input_channels,
        default_sample_rate=float(info.get("defaultSampleRate", 0.0)),
        is_default=_default_input_index(audio_interface) == device_index,
    )


def list_audio_input_devices(audio_interface):
    """Return every input-capable PyAudio device in stable index order."""

    devices = []
    default_index = _default_input_index(audio_interface)
    for index in range(audio_interface.get_device_count()):
        info = audio_interface.get_device_info_by_index(index)
        max_input_channels = int(info.get("maxInputChannels", 0))
        if max_input_channels < 1:
            continue
        devices.append(
            AudioInputDevice(
                index=index,
                name=str(info.get("name", f"device {index}")),
                max_input_channels=max_input_channels,
                default_sample_rate=float(info.get("defaultSampleRate", 0.0)),
                is_default=index == default_index,
            )
        )
    return devices


def _default_input_index(audio_interface):
    try:
        return int(audio_interface.get_default_input_device_info().get("index", -1))
    except (IOError, OSError, ValueError, TypeError):
        return -1


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
