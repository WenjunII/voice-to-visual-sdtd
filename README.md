# Voice-to-Visual Pipeline for StreamDiffusionTD

A real-time bridge between spoken language and high-speed generative visuals. This project uses **OpenAI Whisper** for transcription and a multi-stage **LLM Orchestrator** to transform continuous speech into optimized SDXL prompts for **StreamDiffusionTD**.

## 🚀 Features

- **Live Transcription**: High-performance audio-to-text using **openai-whisper** (Medium model).
- **Intelligent Rephrasing**: Multi-LLM sequence to clean, augment, and structure prompts.
- **Local LLM Sequence**: 
    1. Gemini 3 Flash (via Ollama)
    2. Kimi K2.6 Cloud (via Ollama)
    3. Gemma 4 31B Cloud (via Ollama)
- **Real-time Integration**: Seamless OSC communication with TouchDesigner.
- **Visual Continuity**: LLM-driven scene context management to ensure smooth visual transitions.

## 🛠️ Tech Stack

- **Transcription**: `faster-whisper`
- **LLM Orchestration**: Capriole API (Claude/Gemini/GPT) + Local Ollama
- **Visual Engine**: StreamDiffusion (SDXL-Turbo/Lightning)
- **Bridge**: TouchDesigner (via OSC)
- **Language**: Python 3.10+

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
    Set your Capriole API key in your environment variables:
    ```powershell
    $env:CAPRIOLE_API_KEY = "your_key_here"
    ```

## 🎮 Usage

1.  **Open TouchDesigner**: Load your StreamDiffusionTD project and ensure the OSC In DAT is listening on **Port 7000**.
2.  **Start the Pipeline**:
    ```bash
    python transcriber.py
    ```
3.  **Speak**: The system will automatically capture your speech, refine it, and update the visuals in real-time.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
