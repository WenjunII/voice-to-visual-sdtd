BACKEND_REQUIREMENT_PROFILES = {
    "whisper": "requirements/whisper.txt",
    "faster_whisper": "requirements/faster-whisper.txt",
    "groq": "requirements/groq.txt",
    "groq_hybrid": "requirements/groq-hybrid.txt",
    "google": "requirements/google.txt",
}


def requirement_profile_for_backend(backend):
    try:
        return BACKEND_REQUIREMENT_PROFILES[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported transcription backend: {backend}") from exc


def install_command_for_backend(backend):
    profile = requirement_profile_for_backend(backend)
    return f"python -m pip install -r {profile}"
