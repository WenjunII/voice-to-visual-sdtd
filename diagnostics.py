import importlib.util
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from audio_runtime import get_audio_input_device
from dependency_profiles import install_command_for_backend
from transcription_backends import required_modules_for_backend


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    status: str
    detail: str


def run_diagnostics(
    config,
    sample_rate,
    chunk_size,
    device=None,
):
    backend = config.transcription_backend
    install_command = install_command_for_backend(backend)
    effective_device = device or config.whisper_device
    results = [
        DiagnosticResult("Configuration", "PASS", "validated"),
        DiagnosticResult("Dependency profile", "INFO", install_command),
    ]
    results.append(_python_result())
    results.extend(
        _cuda_results(
            effective_device,
            required=backend in {"whisper", "faster_whisper"},
            install_command=install_command,
        )
    )
    results.extend(_package_results(backend))
    results.append(
        _microphone_result(
            sample_rate,
            chunk_size,
            config.audio_input_device_index,
        )
    )
    results.append(
        _udp_bind_result(config.osc_control_ip, config.osc_control_port)
    )
    results.append(_model_cache_result(config.whisper_model_size))
    results.append(_prompt_budget_result(config))
    results.append(
        _credential_result(
            "Groq credential",
            config.is_secret_configured(config.groq_api_key),
        )
    )
    results.append(
        _credential_result(
            "Capriole credential",
            config.is_secret_configured(config.capriole_api_key),
        )
    )
    results.append(_env_ignore_result())

    print("\n" + "=" * 72)
    print("VOICE-TO-VISUAL STARTUP DIAGNOSTICS")
    print("=" * 72)
    for result in results:
        print(f"[{result.status:<4}] {result.name:<24} {result.detail}")
    print("=" * 72)

    failures = [result for result in results if result.status == "FAIL"]
    if failures:
        print(f"Diagnostics completed with {len(failures)} required failure(s).")
        return 1
    print("Diagnostics completed without required failures.")
    return 0


def _python_result():
    version = ".".join(str(part) for part in sys.version_info[:3])
    status = "PASS" if sys.version_info >= (3, 10) else "FAIL"
    return DiagnosticResult("Python", status, version)


def _cuda_results(device, required, install_command=None):
    if importlib.util.find_spec("torch") is None:
        status = "FAIL" if required else "INFO"
        detail = "not installed"
        if required and install_command:
            detail += f"; run {install_command}"
        return [DiagnosticResult("PyTorch", status, detail)]
    if not required:
        try:
            version = metadata.version("torch")
        except (metadata.PackageNotFoundError, ValueError):
            version = "installed"
        return [
            DiagnosticResult(
                "PyTorch",
                "INFO",
                f"{version}; not loaded for the online backend",
            ),
            DiagnosticResult(
                "CUDA GPU",
                "INFO",
                "not checked because a local CUDA backend is not selected",
            ),
        ]
    try:
        import torch

        results = [DiagnosticResult("PyTorch", "PASS", torch.__version__)]
        available = torch.cuda.is_available()
        cuda_required = required and str(device).startswith("cuda")
        status = "PASS" if available else ("FAIL" if cuda_required else "INFO")
        detail = torch.cuda.get_device_name(0) if available else "CUDA unavailable"
        results.append(DiagnosticResult("CUDA GPU", status, detail))
        if available:
            results.append(
                DiagnosticResult(
                    "CUDA / cuDNN",
                    "PASS",
                    f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}",
                )
            )
        return results
    except Exception as exc:
        status = "FAIL" if required else "INFO"
        return [DiagnosticResult("PyTorch", status, str(exc))]


def _package_results(backend):
    required_modules = required_modules_for_backend(backend)
    install_command = install_command_for_backend(backend)
    packages = {
        "python-osc": ("pythonosc", True),
        "PyAudio": ("pyaudio", True),
        "Silero VAD": ("silero_vad", False),
        "Transformers": ("transformers", False),
        "OpenAI Whisper": ("whisper", "whisper" in required_modules),
        "faster-whisper": (
            "faster_whisper",
            "faster_whisper" in required_modules,
        ),
        "CTranslate2": ("ctranslate2", "ctranslate2" in required_modules),
        "SpeechRecognition": (
            "speech_recognition",
            "speech_recognition" in required_modules,
        ),
        "Argos Translate": (
            "argostranslate",
            "argostranslate" in required_modules,
        ),
        "LangDetect": ("langdetect", "langdetect" in required_modules),
    }
    results = []
    for label, (module, required) in packages.items():
        installed = importlib.util.find_spec(module) is not None
        status = "PASS" if installed else ("FAIL" if required else "INFO")
        if installed:
            detail = "installed"
        elif required:
            detail = f"not installed; run {install_command}"
        else:
            detail = "not installed (optional)"
        results.append(DiagnosticResult(label, status, detail))
    return results


def _microphone_result(sample_rate, chunk_size, input_device_index=None):
    audio = None
    stream = None
    try:
        import pyaudio

        audio = pyaudio.PyAudio()
        device = get_audio_input_device(audio, input_device_index)
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=chunk_size,
        )
        return DiagnosticResult(
            "Microphone",
            "PASS",
            f"[{device.index}] {device.name}",
        )
    except Exception as exc:
        return DiagnosticResult("Microphone", "FAIL", str(exc))
    finally:
        if stream is not None:
            stream.close()
        if audio is not None:
            audio.terminate()


def _udp_bind_result(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((ip, port))
        return DiagnosticResult("OSC control port", "PASS", f"{ip}:{port} available")
    except OSError as exc:
        return DiagnosticResult("OSC control port", "WARN", f"{ip}:{port} unavailable ({exc})")
    finally:
        sock.close()


def _model_cache_result(model_size):
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_root = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    model_dir = cache_root / f"models--Systran--faster-whisper-{model_size}"
    if model_dir.exists():
        return DiagnosticResult("Whisper model cache", "PASS", str(model_dir))
    openai_model = Path.home() / ".cache" / "whisper" / f"{model_size}.pt"
    if openai_model.exists():
        return DiagnosticResult("Whisper model cache", "PASS", str(openai_model))
    return DiagnosticResult("Whisper model cache", "WARN", f"{model_size} will download on first use")


def _prompt_budget_result(config, tokenizer_loader=None):
    if not config.prompt_token_budget_enabled:
        return DiagnosticResult(
            "Prompt tokenizers",
            "INFO",
            "budgeting disabled",
        )

    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer

            tokenizer_loader = AutoTokenizer.from_pretrained
        except Exception:
            tokenizer_loader = None

    missing = []
    if tokenizer_loader is None:
        missing = list(config.prompt_tokenizer_models)
    else:
        for model in config.prompt_tokenizer_models:
            try:
                tokenizer_loader(model, local_files_only=True)
            except Exception:
                missing.append(model)

    total = len(config.prompt_tokenizer_models)
    available = total - len(missing)
    if not missing and total:
        return DiagnosticResult(
            "Prompt tokenizers",
            "PASS",
            f"{available}/{total} available locally; exact budgeting ready",
        )

    if config.prompt_token_budget_fallback == "conservative":
        return DiagnosticResult(
            "Prompt tokenizers",
            "WARN",
            f"{available}/{total} available locally; conservative offline "
            "budgeting will be used",
        )
    return DiagnosticResult(
        "Prompt tokenizers",
        "FAIL",
        f"{available}/{total} available locally and fallback is off",
    )


def _credential_result(name, configured):
    return DiagnosticResult(name, "PASS" if configured else "INFO", "configured locally" if configured else "not configured")


def _env_ignore_result():
    try:
        result = subprocess.run(
            ["git", "check-ignore", ".env"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return DiagnosticResult("Credential file", "PASS", ".env is ignored by Git")
        return DiagnosticResult("Credential file", "FAIL", ".env is not ignored by Git")
    except Exception as exc:
        return DiagnosticResult("Credential file", "WARN", f"could not check Git ignore ({exc})")
