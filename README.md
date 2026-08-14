# Voice-to-Visual Pipeline for StreamDiffusionTD

A real-time bridge between spoken language and high-speed generative visuals. The project supports optimized local GPU Whisper, the original OpenAI Whisper runtime, and hosted transcription backends, then turns stable speech updates into SDXL prompts for **StreamDiffusionTD**.

## Features

- **Cultural Fusion Generation**: Fixed Prompt Template with live visual identity modes for Asian-American visuals, Black and Brown people visuals, or combined Asian + Black and Brown visuals.
- **Live Prompt Style Selection**: Switch between the original human figure focus and a general scene template with no central human figure.
- **Live Gender, Age & Visual Identity Selection**: Interactive keyboard and OSC controls toggle the subject's identity (Man/Woman, Young/Adult/Elder), prompt style, and visual representation mode. Visual changes immediately rebuild the active prompt without waiting for more speech.
- **Stable Streaming Transcription**: Confirms the word prefix shared by consecutive hypotheses, reducing repeated text and prompt flicker.
- **Conservative Hallucination Filtering**: Removes standalone Whisper outro artifacts such as “thanks for watching” without discarding real sentences that merely contain similar words.
- **Rolling Scene Memory**: Removes overlap between audio segments and carries the newest subjects, places, and actions across prompt updates.
- **Live Transcription**: Selectable audio-to-text using optimized local GPU **faster-whisper**, the original local GPU **OpenAI Whisper**, online **Groq Whisper** translation, or an experimental Groq turbo + local CPU translation hybrid.
- **Multilingual Translation**: Automatically translates Chinese, Cantonese, Spanish, and other languages into English in real-time, allowing non-English speakers to control the visual engine seamlessly.
- **CPU Voice Activity Detection (VAD)**: Silero VAD keeps quiet phonemes, natural pauses, and audio pre-roll without consuming StreamDiffusion's GPU memory. Energy detection remains an automatic fallback.
- **Automatic Microphone Recovery**: Brief read glitches retry in place, repeated failures safely finalize buffered speech, and disconnected or unavailable devices reopen with capped exponential backoff.
- **Explicit Microphone Selection**: List available input devices and pin live capture and diagnostics to a specific PyAudio device index, or keep following the Windows system default.
- **Audio Source Adapters & WAV Replay**: Keeps microphone ownership outside the pipeline and can replay a recording through the real VAD, segmentation, scheduling, transcription, prompt, logging, and OSC path without audio hardware.
- **Bounded Audio Segments**: Long speech is split into configurable segments with overlap so words at a boundary are less likely to disappear.
- **Backpressure-Aware Scheduling**: Final speech is prioritized in a bounded queue while obsolete partial snapshots are replaced, preventing latency from growing during continuous speech.
- **Retry-Aware Online Transcription**: Transient Groq and Google failures preserve final segments for bounded retries and respect Groq's `Retry-After` response header.
- **Isolated Backend Adapters**: Local Whisper, faster-whisper, Groq, hybrid translation, and Google each own their model or API contract, timing limits, and resource cleanup behind one runtime interface.
- **Exact SDXL Prompt Budgeting**: Checks both SDXL CLIP tokenizers and switches to compact prompt wording when necessary so prompts stay inside the 77-token context window.
- **Two-Way OSC Integration**: Sends prompts and runtime health to TouchDesigner on port 7000 and accepts live controls from TouchDesigner on port 7001.
- **Validated Runtime Configuration**: Types and checks environment settings before startup, reports every configuration problem together, and safely shows effective values with credentials redacted.
- **Instance-Scoped Runtime Settings**: Every pipeline uses its own immutable configuration for audio, VAD, scheduling, prompts, retries, and OSC, preventing settings from leaking between embedded or test instances.
- **Structured Session Logging**: Labels operational events by subsystem, captures latency/retry/reconnection metrics, optionally rotates JSON Lines log files, and redacts configured credentials.
- **Graceful Worker Shutdown**: Uses cooperative cancellation, owned non-daemon workers, interruptible retry waits, and ordered cleanup so backend and log resources stay open until audio and transcription stop.
- **Startup Diagnostics**: Checks CUDA, cuDNN, the microphone, packages, model cache, OSC input port, local credential presence, and `.env` Git-ignore status without printing secrets.

## Tech Stack

- **Transcription**: `faster-whisper` (recommended Small model on CUDA), `openai-whisper` (preserved local backend), Groq `whisper-large-v3` translation, Groq `whisper-large-v3-turbo` transcription with local Argos Translate, or `SpeechRecognition` with Google Speech Recognition (recognition-only experiment)
- **Optional LLM Orchestration**: `orchestrator.py` can refine a single prompt through its configured Ollama model sequence. The live `transcriber.py` path uses the fixed prompt templates directly.
- **Visual Engine**: StreamDiffusion (SDXL-Turbo/Lightning)
- **Bridge**: TouchDesigner via OSC output on port 7000 and control input on port 7001
- **Language**: Python 3.10+

## Fixed Prompt Strategy

The system keeps the original human-focused template and adds a second scene-focused template for moments when you want the visuals to describe a place, mood, or environment instead of centering a person.

Both original templates remain available as the high-detail variants. When a complete prompt would exceed SDXL's context window, the bridge automatically uses a compact equivalent and retains the newest transcript details. Token limits are checked against both SDXL text encoders.

Human figure focus:

```text
A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age} {gender}, 
{visual identity context}, 8k UHD, highly detailed...
```

General scene:

```text
A hyper-realistic photorealistic cinematic scene of {text}, {scene identity context},
environment-focused composition, no central human figure, no portrait framing, 8k UHD, highly detailed...
```

## Interactive Controls

While `transcriber.py` is running, you can use the following keyboard shortcuts to adjust the visuals live:

| Category | Key | Action |
| :--- | :--- | :--- |
| **Gender** | `m` | Set focus to **Man** |
| | `w` | Set focus to **Woman** |
| | `n` | Set focus to **Neutral/Person** |
| **Age** | `1` | Set focus to **Young** |
| | `2` | Set focus to **Adult** |
| | `3` | Set focus to **Elderly** |
| **Visual Mode** | `d` | Use default **Asian-American** visuals |
| | `b` | Use **Black and Brown people** visuals |
| | `x` | Use **Asian + Black and Brown people** visuals |
| **Prompt Style** | `f` | Use original **Human Figure** focus |
| | `g` | Use **General Scene** focus with no central human figure |
| **Language** | `e` | Force **English** transcription |
| | `c` | Force **Chinese** (Mandarin/Cantonese) |
| | `s` | Force **Spanish** transcription |
| | `a` | **Auto-detect** language (Default) |

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/wenjunii/voice-to-visual-sdtd.git
    cd voice-to-visual-sdtd
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    Copy the example environment file and keep real credentials in your local `.env` only:

    ```powershell
    Copy-Item .env.example .env
    ```

    The project ignores `.env` and `.env.*` files so API keys are not synced to GitHub. Keep `.env.example` placeholder-only.

    After changing `.env`, validate the effective values before starting the audio or model runtimes:

    ```powershell
    python transcriber.py --check-config
    ```

    The command checks types, supported choices, ports, positive ranges, and related minimum/maximum settings. It prints the effective configuration while replacing configured API keys with `<redacted>`. Invalid settings return exit code `2` with all detected problems in one report.

    Set your Capriole API key (if using cloud models). You can do this in two ways:

    *   **Option A: Use a `.env` file (Recommended)**
        Create a `.env` file in the root directory of the project (which is automatically ignored by Git to keep your key secure):
        ```env
        CAPRIOLE_API_KEY="your_key_here"
        ```

    *   **Option B: Set Environment Variable**
        Or, set it directly in your terminal environment:
        ```powershell
        $env:CAPRIOLE_API_KEY = "your_key_here"
        ```

    Optional live transcription settings:

    ```env
    # Recommended default: optimized local Whisper translation on a CUDA GPU.
    TRANSCRIPTION_BACKEND=faster_whisper
    WHISPER_MODEL_SIZE=small
    WHISPER_DEVICE=cuda
    # faster-whisper runs on the GPU while int8_float16 reduces its memory footprint.
    FASTER_WHISPER_COMPUTE_TYPE=int8_float16
    FASTER_WHISPER_CPU_THREADS=4
    FASTER_WHISPER_NUM_WORKERS=1
    # Shared local Whisper speed/latency tuning. Use small/base for faster but lower-quality output.
    WHISPER_TRANSCRIPTION_INTERVAL=0.8
    WHISPER_MIN_AUDIO_SECONDS=0.8
    WHISPER_MAX_AUDIO_SECONDS=6.0
    WHISPER_BEAM_SIZE=1
    WHISPER_BEST_OF=1
    WHISPER_TEMPERATURE=0.0
    WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
    WHISPER_LOG_LATENCY=true

    # CPU VAD and stable streaming settings.
    VAD_ENGINE=silero
    VAD_THRESHOLD=0.5
    VAD_ENERGY_THRESHOLD=400
    VAD_PRE_ROLL_SECONDS=0.32
    VAD_SILENCE_SECONDS=0.7
    STREAM_OVERLAP_SECONDS=0.5
    TRANSCRIPT_CONFIRM_UPDATES=2

    # Optional: pin capture to a device reported by --list-audio-devices.
    # Leave unset to follow the Windows system default across reconnects.
    # AUDIO_INPUT_DEVICE_INDEX=2

    # Recover from microphone startup failures and live disconnections.
    AUDIO_RECONNECT_ENABLED=true
    AUDIO_RECONNECT_BASE_SECONDS=0.5
    AUDIO_RECONNECT_MAX_SECONDS=8.0
    AUDIO_MAX_CONSECUTIVE_READ_ERRORS=3
    AUDIO_READ_RETRY_SECONDS=0.1

    # Bounded scheduler: finals have priority and obsolete partials are replaced.
    TRANSCRIPTION_MAX_FINAL_JOBS=8
    TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS=4.0
    TRANSCRIPTION_FINAL_MAX_AGE_SECONDS=30.0
    TRANSCRIPTION_FINAL_MAX_RETRIES=2
    TRANSCRIPTION_RETRY_BASE_SECONDS=1.0
    TRANSCRIPTION_RETRY_MAX_SECONDS=10.0

    # OSC output and optional OSC controls/status for TouchDesigner.
    OSC_IP=127.0.0.1
    OSC_PORT=7000
    OSC_CONTROL_ENABLED=true
    OSC_CONTROL_IP=127.0.0.1
    OSC_CONTROL_PORT=7001
    OSC_STATUS_INTERVAL=0.5

    # Human-readable operational console logs and optional rotating JSON Lines files.
    RUNTIME_LOG_LEVEL=info
    RUNTIME_LOG_CONSOLE_ENABLED=true
    # Leave blank to disable persistent logs.
    RUNTIME_LOG_FILE=
    RUNTIME_LOG_MAX_BYTES=5000000
    RUNTIME_LOG_BACKUP_COUNT=3
    # Log overdue workers after this grace period, then keep waiting before cleanup.
    RUNTIME_SHUTDOWN_GRACE_SECONDS=25.0

    # Keep recent scene details while removing repeated overlap between segments.
    SCENE_MEMORY_MAX_WORDS=36
    SCENE_MEMORY_MAX_AGE_SECONDS=20

    # Enforce both SDXL CLIP encoders' prompt limit.
    PROMPT_TOKEN_BUDGET_ENABLED=true
    PROMPT_MAX_TOKENS=77
    PROMPT_MIN_TRANSCRIPT_TOKENS=20
    PROMPT_LOG_TOKENS=true
    PROMPT_TOKENIZER_MODELS=openai/clip-vit-large-patch14,laion/CLIP-ViT-bigG-14-laion2B-39B-b160k

    # Recommended online option for StreamDiffusion: multilingual audio -> English prompt text.
    # Uses Groq's hosted Whisper translation endpoint. The free plan has rate limits.
    # Requires internet, a free Groq API key, and sends microphone audio to Groq.
    # TRANSCRIPTION_BACKEND=groq
    # GROQ_API_KEY="your_groq_key_here"
    # Faster Groq settings for lower latency. 3.2s stays below the 20 RPM free limit.
    # Groq translation requires whisper-large-v3; turbo is transcription-only.
    # GROQ_TRANSCRIPTION_MODEL=whisper-large-v3
    # GROQ_TEXT_TRANSLATION_MODEL=llama-3.1-8b-instant
    # GROQ_RESPONSE_FORMAT=text
    # GROQ_ENGLISH_FALLBACK=auto
    # GROQ_TRANSCRIPTION_INTERVAL=3.2
    # GROQ_MIN_AUDIO_SECONDS=1.0
    # GROQ_MAX_AUDIO_SECONDS=6.0
    # GROQ_REQUEST_TIMEOUT=20.0
    # GROQ_LOG_LATENCY=true

    # Experimental hybrid mode:
    # Groq turbo transcribes online, then Argos Translate translates non-English text locally on CPU.
    # This may be faster than Groq audio translation, but quality depends on the local translator.
    # Whisper cannot do this local text translation step; Whisper only translates audio.
    # If Argos cannot translate, groq_text optionally provides an online text-only fallback.
    # TRANSCRIPTION_BACKEND=groq_hybrid
    # GROQ_HYBRID_MODEL=whisper-large-v3-turbo
    # LOCAL_TRANSLATOR=argos
    # LOCAL_TRANSLATOR_TARGET_LANGUAGE=en
    # LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE=zh
    # LOCAL_TRANSLATOR_PRELOAD_LANGUAGES=zh,es
    # LOCAL_TRANSLATOR_AUTO_INSTALL=true
    # LOCAL_TRANSLATOR_LOG_LATENCY=true
    # Use groq_text for fallback, or off to keep translation local-only.
    # HYBRID_TRANSLATION_FALLBACK=groq_text

    # Published base limits for both Groq Whisper models:
    # 20 requests/minute, 2,000 requests/day,
    # 7,200 audio seconds/hour, 28,800 audio seconds/day.
    # Your Groq Console Limits page is authoritative for your organization.

    # Recognition-only online experiment. This does not translate to English.
    # TRANSCRIPTION_BACKEND=google
    # GOOGLE_SPEECH_LANGUAGE=en-US
    # GOOGLE_SPEECH_CHINESE_LANGUAGE=zh-CN
    # GOOGLE_SPEECH_SPANISH_LANGUAGE=es-ES
    ```

    Groq documents the same base speech limits for `whisper-large-v3` and `whisper-large-v3-turbo`, but limits apply at the organization level. Check the [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits) and your Console Limits page for the current values assigned to your account.

## Usage

1.  **Validate configuration after editing `.env`**:
    ```powershell
    python transcriber.py --check-config
    ```
    Command-line `--backend` and `--input-device` overrides are included in the displayed effective configuration.
2.  **Check the machine once after setup or dependency changes**:
    ```powershell
    python transcriber.py --diagnose
    ```
    Diagnostics start from the same validated configuration and report only whether credentials are configured; they never print credential values.
3.  **Open TouchDesigner**: Load your StreamDiffusionTD project and ensure the OSC In DAT is listening on **Port 7000**.
4.  **Choose a microphone when needed**: The system default is used automatically. On a multi-device installation, list available inputs and select one for the current run:
    ```powershell
    python transcriber.py --list-audio-devices
    python transcriber.py --input-device 2
    ```
    To keep the selection across runs, set `AUDIO_INPUT_DEVICE_INDEX=2` in `.env`. Diagnostics use the same selected device.
5.  **Start the Pipeline with your `.env` default**:
    ```bash
    python transcriber.py
    ```
    To choose a backend for just one run without editing `.env`:
    ```powershell
    python transcriber.py --backend faster_whisper
    python transcriber.py --backend whisper
    python transcriber.py --backend groq
    python transcriber.py --backend groq_hybrid
    ```
    `faster_whisper` uses CTranslate2 on the CUDA GPU and is the recommended local mode when StreamDiffusion shares the same GPU. `whisper` preserves the original OpenAI Whisper CUDA implementation. Both auto-detect multilingual audio and use Whisper's `translate` task to produce English. `groq` uses online multilingual translation and does not load local Whisper. `groq_hybrid` uses Groq turbo for online transcription, then translates non-English text locally on CPU with Argos Translate when available. If local translation is unavailable or still returns Chinese/Cantonese text, `HYBRID_TRANSLATION_FALLBACK=groq_text` sends only the transcript text through a fast Groq chat model for English cleanup.

    The dependency file pins CTranslate2 `4.4.0` for this project's current Windows CUDA 12 + cuDNN 8 setup. Silero VAD runs on the CPU. If Silero cannot load, the script reports the problem and automatically uses the energy detector.

    The first `faster_whisper` run downloads and caches its converted Whisper model. Prompt budgeting also caches two small tokenizer configurations. Later launches reuse both local caches.

    You can also override the backend for the current PowerShell session:
    ```powershell
    $env:TRANSCRIPTION_BACKEND = "groq"
    python transcriber.py
    ```
6.  **Speak & Control**: The system will automatically capture your speech. Use the keys above or the OSC controls below to change the visuals as you talk.

### Deterministic WAV Replay

Replay an uncompressed 16-bit PCM WAV recording through the live pipeline at real-time speed:

```powershell
python transcriber.py --replay .\sample.wav
python transcriber.py --backend faster_whisper --replay .\sample.wav
```

Replay accepts mono or multichannel PCM input at any valid sample rate. It converts the recording to the runtime's 16 kHz mono format, sends stable chunks through the same VAD and scheduler used by the microphone, drains the final transcription job, and then exits automatically. OSC output and controls remain enabled, so a recording can reproduce a complete TouchDesigner session without PyAudio or a live input device.

`--replay` cannot be combined with `--benchmark` or `--input-device`. Benchmark mode calls the selected local model repeatedly to measure performance; replay mode processes the recording once through the complete streaming system.

### Runtime Logging

Operational events use `debug`, `info`, `warning`, `error`, or `critical` levels and identify their subsystem, including `audio`, `backend`, `transcription`, `scheduler`, `prompt`, `osc`, and `control`. Console logs remain human-readable. Set `RUNTIME_LOG_FILE` to enable persistent JSON Lines logs:

```env
RUNTIME_LOG_LEVEL=info
RUNTIME_LOG_CONSOLE_ENABLED=true
RUNTIME_LOG_FILE=logs/voice-to-visual.jsonl
RUNTIME_LOG_MAX_BYTES=5000000
RUNTIME_LOG_BACKUP_COUNT=3
RUNTIME_SHUTDOWN_GRACE_SECONDS=25.0
```

The file rotates before exceeding the configured size and retains the configured number of backups. Each record contains a timestamp, session ID, subsystem, event name, level, message, and relevant metrics. Raw prompt and transcript text remains in the live console instead of the operational file, and configured API keys plus bearer credentials are replaced with `<redacted>`.

### Graceful Shutdown

Ctrl+C and terminal audio failures signal the audio and transcription workers through a shared cancellation event. Retry and idle waits wake immediately, the OSC control server stops accepting changes, and the runtime waits for in-flight work before closing the transcription backend and log session.

`RUNTIME_SHUTDOWN_GRACE_SECONDS` is an observability threshold rather than a destructive timeout. If a worker is still active after the grace period, the runtime records a `worker_shutdown_overdue` warning with the worker name and continues waiting. This preserves cleanup ordering and avoids closing an HTTP session, model, or log handler while a worker is still using it.

### OSC Output to TouchDesigner

| Address | Value |
| :--- | :--- |
| `/prompt` | Complete SDXL prompt |
| `/partial_text` | Current stable transcript |
| `/scene_context` | Rolling merged scene text |
| `/prompt_tokens` | Maximum token count across both SDXL text encoders |
| `/transcript_final` | Finalized transcript segment |
| `/backend_status` | `ready`, `transcribing`, `retrying`, `error`, or `stopped` |
| `/backend` | Active transcription backend |
| `/is_speaking` | `1` while VAD detects an active speech segment, otherwise `0` |
| `/queue_depth` | Pending final jobs plus the newest pending partial |
| `/latency_total` | Seconds from scheduler submission through completed transcription |
| `/latency_asr` | Seconds spent in the active ASR/translation call |
| `/retry_in` | Seconds until a rate-limited or transiently unavailable backend may be called again |
| `/dropped_jobs` | Total stale or capacity-dropped jobs |
| `/audio_status` | `starting`, `ready`, `degraded`, `reconnecting`, `error`, or `stopped` |
| `/audio_source` | `microphone` or `wav_replay` |
| `/audio_reconnects` | Total microphone reopen attempts during this run |
| `/audio_error` | Most recent microphone error, cleared after recovery |
| `/audio_device_index` | Resolved PyAudio input-device index, or `-1` for unresolved/default input and WAV replay |
| `/audio_device_name` | Resolved microphone name or replay filename |
| `/gender` | Active gender mode |
| `/age` | Active age mode |
| `/visual_mode` | Active visual identity mode |
| `/prompt_style` | Active human-focus or general-scene mode |
| `/language` | Active language code or `auto` |

### OSC Input from TouchDesigner

Send these messages to `127.0.0.1:7001`. Text values accept the names below or the equivalent keyboard key.

| Address | Accepted values |
| :--- | :--- |
| `/control/gender` | `man`, `woman`, `neutral` |
| `/control/age` | `young`, `adult`, `elder` |
| `/control/visual_mode` | `asian_american`, `black_brown`, `asian_black_brown` |
| `/control/prompt_style` | `human_focus`, `general_scene` |
| `/control/language` | `en`, `zh`, `es`, `auto` |
| `/control/reset_scene` | Any value; clears rolling scene memory |
| `/control/request_status` | Any value; immediately emits all runtime status addresses |

Accepted changes return `/control_ack`; scene resets additionally emit `/scene_reset`. Gender, age, visual-mode, and prompt-style changes immediately resend `/prompt` using the current scene context. `/control/request_status` includes all five active mode addresses so a TouchDesigner interface can resynchronize after either process restarts.

## Runtime Structure

- `transcriber.py` coordinates live audio, scheduling, prompt generation, controls, cooperative worker cancellation, and ordered process cleanup. It loads `.env` only when constructing a default pipeline or entering the CLI, so importing the module does not parse, validate, or cache project runtime settings.
- `audio_sources.py` owns lazy PyAudio setup, microphone streams, WAV decoding/resampling, deterministic chunking, replay pacing, and source cleanup.
- `transcription_backends.py` owns lazy model/API setup, backend-specific transcription and translation contracts, audio timing limits, and resource cleanup.
- `runtime_config.py` loads, types, validates, and safely reports environment-backed configuration.
- `runtime_logging.py` owns session IDs, subsystem context, credential redaction, human console formatting, and rotating JSON Lines files.
- `audio_runtime.py` owns CPU voice activity detectors.
- `runtime_scheduler.py` owns bounded final/partial scheduling and queue metrics.
- `backend_errors.py` owns retry timing and `Retry-After` parsing.
- `osc_control.py` owns the TouchDesigner control server and control aliases.
- `diagnostics.py` owns the read-only startup health checks.
- `transcript_filter.py` conservatively removes known standalone Whisper hallucinations.
- `streaming_core.py` and `prompt_engine.py` own testable speech segmentation, transcript stability, scene memory, and prompt budgeting.

## Local Performance Benchmark

Compare the two local GPU engines with the same 16-bit PCM WAV recording:

```powershell
python transcriber.py --backend faster_whisper --benchmark .\sample.wav --benchmark-runs 3
python transcriber.py --backend whisper --benchmark .\sample.wav --benchmark-runs 3
```

The benchmark shares replay's validated PCM decoder, performs one warm-up pass, then reports latency and real-time factor for each measured run. A real-time factor below `1.0` means transcription is faster than the recording duration.

Run the streaming logic tests without loading a Whisper model:

```powershell
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Pull requests and updates to `main` run the same unit suite on Windows with Python 3.10 and 3.11. The lightweight test requirements omit CUDA, Whisper, PyAudio, and StreamDiffusion because those hardware integrations are mocked in unit tests. The suite also verifies configuration isolation, side-effect-free imports, WAV conversion and replay, microphone adapter cleanup and recovery, log rotation, credential redaction, interruptible cancellation, worker crashes, and ordered shutdown. `python transcriber.py --diagnose` remains available even when PyAudio is missing, so a new setup can report the missing microphone dependency instead of failing during import.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
