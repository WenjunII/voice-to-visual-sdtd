# Voice-to-Visual Pipeline for StreamDiffusionTD

A real-time bridge between spoken language and high-speed generative visuals. This project uses **OpenAI Whisper** for transcription and a multi-stage **LLM Orchestrator** to transform continuous speech into optimized SDXL prompts for **StreamDiffusionTD**.

## 🚀 Features

- **Cultural Fusion Generation**: Fixed Prompt Template optimized for the Asian-American experience, blending modern US settings with traditional Chinese motifs.
- **Live Transcription**: High-performance audio-to-text using **OpenAI Whisper** (Medium model).
- **Multi-Stage Orchestration**: Fallback chain for prompt refinement ensuring high-quality visual descriptors even when primary models are unavailable.
- **Voice Activity Detection (VAD)**: Smart volume gating and a 5-second auto-reset timer to prevent "ghost" transcriptions and hallucinations.
- **Token Management**: 12-second rolling buffer to ensure prompts stay within SDXL's 77-token limit.
- **Multi-Threaded Architecture**: Concurrent audio capture and transcription for fluid, non-blocking performance.
- **Real-time Integration**: Ultra-fast OSC updates to TouchDesigner for instantaneous visual feedback.

## 🛠️ Tech Stack

- **Transcription**: `openai-whisper` (Medium Model)
- **LLM Orchestration**: 
    1. **Gemini 3 Flash Preview** (Primary)
    2. **Kimi k2.6** (Fallback 1)
    3. **Gemma 4 (31b)** (Fallback 2)
- **Visual Engine**: StreamDiffusion (SDXL-Turbo/Lightning)
- **Bridge**: TouchDesigner (via OSC on Port 7000)
- **Language**: Python 3.10+

## 🎨 Fixed Prompt Strategy

The system utilizes a specialized template to maintain a consistent "Cultural Fusion" aesthetic:

```text
A hyper-realistic photorealistic cinematic shot of {text}, capturing the Chinese-American identity, 
blending modern US urban settings with subtle traditional Chinese cultural motifs and textures, 
8k UHD, highly detailed, masterfully lit, fusion of East and West aesthetics...
```

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
    Set your Capriole API key (if using cloud models) in your environment variables:
    ```powershell
    $env:CAPRIOLE_API_KEY = "your_key_here"
    ```

## 🎮 Usage

1.  **Open TouchDesigner**: Load your StreamDiffusionTD project and ensure the OSC In DAT is listening on **Port 7000**.
2.  **Start the Pipeline**:
    ```bash
    python transcriber.py
    ```
3.  **Speak**: The system will automatically capture your speech, refine it via the LLM chain, and update the visuals in real-time.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

