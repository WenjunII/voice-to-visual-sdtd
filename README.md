# Voice-to-Visual Pipeline for StreamDiffusionTD

A real-time bridge between spoken language and high-speed generative visuals. This project uses **OpenAI Whisper** for transcription and a multi-stage **LLM Orchestrator** to transform continuous speech into optimized SDXL prompts for **StreamDiffusionTD**.

## 🚀 Features

- **Cultural Fusion Generation**: Fixed Prompt Template with live visual identity modes for Asian-American visuals, Black and Brown people visuals, or combined Asian + Black and Brown visuals.
- **Live Prompt Style Selection**: Switch between the original human figure focus and a general scene template with no central human figure.
- **Live Gender, Age & Visual Identity Selection**: Interactive keyboard controls to toggle the subject's identity (Man/Woman, Young/Adult/Elder), prompt style, and visual representation mode in real-time.
- **Responsive Prompt Reversal**: Automatically reverses the order of spoken sentences so the **most recent speech** is placed at the start of the prompt for immediate visual feedback.
- **Live Transcription**: Selectable audio-to-text using local GPU **OpenAI Whisper** (Medium model), online **Groq Whisper** translation, or an experimental Groq turbo + local CPU translation hybrid.
- **Multilingual Translation**: Automatically translates Chinese, Cantonese, Spanish, and other languages into English in real-time, allowing non-English speakers to control the visual engine seamlessly.
- **Voice Activity Detection (VAD)**: Smart volume gating and a 5-second auto-reset timer to prevent "ghost" transcriptions and hallucinations.
- **Token Management**: 12-second rolling buffer to ensure prompts stay within SDXL's 77-token limit.
- **Real-time Integration**: Ultra-fast OSC updates to TouchDesigner for instantaneous visual feedback.

## 🛠️ Tech Stack

- **Transcription**: `openai-whisper` (Medium Model, CUDA GPU default), Groq `whisper-large-v3` translation, Groq `whisper-large-v3-turbo` transcription with local Argos Translate, or `SpeechRecognition` with Google Speech Recognition (recognition-only experiment)
- **LLM Orchestration**: 
    1. **Gemini 3 Flash Preview** (Primary)
    2. **Kimi k2.6** (Fallback 1)
    3. **Gemma 4 (31b)** (Fallback 2)
- **Visual Engine**: StreamDiffusion (SDXL-Turbo/Lightning)
- **Bridge**: TouchDesigner (via OSC on Port 7000)
- **Language**: Python 3.10+

## 🎨 Fixed Prompt Strategy

The system keeps the original human-focused template and adds a second scene-focused template for moments when you want the visuals to describe a place, mood, or environment instead of centering a person.

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

## 🎮 Interactive Controls

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

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/WenjunII/voice-to-visual-sdtd.git
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
    # Default: local Whisper, best multilingual translation, requires a CUDA GPU.
    TRANSCRIPTION_BACKEND=whisper
    WHISPER_MODEL_SIZE=medium
    WHISPER_DEVICE=cuda
    # Local Whisper speed/latency tuning. Use small/base for faster but lower-quality output.
    WHISPER_TRANSCRIPTION_INTERVAL=0.8
    WHISPER_MIN_AUDIO_SECONDS=0.8
    WHISPER_MAX_AUDIO_SECONDS=6.0
    WHISPER_BEAM_SIZE=1
    WHISPER_BEST_OF=1
    WHISPER_TEMPERATURE=0.0
    WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
    WHISPER_LOG_LATENCY=true

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
    # GROQ_LOG_LATENCY=true

    # Experimental hybrid mode:
    # Groq turbo transcribes online, then Argos Translate translates non-English text locally on CPU.
    # This may be faster than Groq audio translation, but quality depends on the local translator.
    # Whisper cannot do this local text translation step; Whisper only translates audio.
    # If Argos is not installed or cannot load, hybrid mode still transcribes but passes non-English text through.
    # TRANSCRIPTION_BACKEND=groq_hybrid
    # GROQ_HYBRID_MODEL=whisper-large-v3-turbo
    # LOCAL_TRANSLATOR=argos
    # LOCAL_TRANSLATOR_TARGET_LANGUAGE=en
    # LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE=zh
    # LOCAL_TRANSLATOR_PRELOAD_LANGUAGES=zh,es
    # LOCAL_TRANSLATOR_AUTO_INSTALL=true
    # LOCAL_TRANSLATOR_LOG_LATENCY=true
    # HYBRID_TRANSLATION_FALLBACK=groq_text

    # Current Groq free-plan limits for whisper-large-v3:
    # 20 requests/minute, 2,000 requests/day,
    # 7,200 audio seconds/hour, 28,800 audio seconds/day.

    # Recognition-only online experiment. This does not translate to English.
    # TRANSCRIPTION_BACKEND=google
    # GOOGLE_SPEECH_LANGUAGE=en-US
    # GOOGLE_SPEECH_CHINESE_LANGUAGE=zh-CN
    # GOOGLE_SPEECH_SPANISH_LANGUAGE=es-ES
    ```

## 🕹️ Usage

1.  **Open TouchDesigner**: Load your StreamDiffusionTD project and ensure the OSC In DAT is listening on **Port 7000**.
2.  **Start the Pipeline with your `.env` default**:
    ```bash
    python transcriber.py
    ```
    To choose a backend for just one run without editing `.env`:
    ```powershell
    python transcriber.py --backend groq
    python transcriber.py --backend groq_hybrid
    python transcriber.py --backend whisper
    ```
    `groq` uses online multilingual translation and does not load local Whisper. `groq_hybrid` uses Groq turbo for online transcription, then translates non-English text locally on CPU with Argos Translate when available. If local translation is unavailable or still returns Chinese/Cantonese text, `HYBRID_TRANSLATION_FALLBACK=groq_text` sends only the transcript text through a fast Groq chat model for English cleanup. `whisper` uses local OpenAI Whisper on CUDA GPU. If Groq audio translation returns Chinese/Cantonese text, `GROQ_ENGLISH_FALLBACK=auto` sends that text through a fast Groq chat model for an English prompt. For the lowest latency, local Whisper is usually better because it avoids network round trips. You can make local Whisper faster by lowering `WHISPER_MODEL_SIZE` to `small` or `base`, reducing `WHISPER_MAX_AUDIO_SECONDS`, or increasing `WHISPER_TRANSCRIPTION_INTERVAL`.

    You can also override the backend for the current PowerShell session:
    ```powershell
    $env:TRANSCRIPTION_BACKEND = "groq"
    python transcriber.py
    ```
3.  **Speak & Control**: The system will automatically capture your speech. Use the keys above to shift the identity of the generated figures as you talk.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
