# Voice-to-Visual Pipeline for StreamDiffusionTD

A real-time bridge between spoken language and high-speed generative visuals. This project uses **OpenAI Whisper** for transcription and a multi-stage **LLM Orchestrator** to transform continuous speech into optimized SDXL prompts for **StreamDiffusionTD**.

## 🚀 Features

- **Cultural Fusion Generation**: Fixed Prompt Template optimized for the Asian-American experience, blending modern US settings with traditional Chinese motifs.
- **Live Gender & Age Selection**: Interactive keyboard controls to toggle the subject's identity (Man/Woman, Young/Adult/Elder) in real-time.
- **Responsive Prompt Reversal**: Automatically reverses the order of spoken sentences so the **most recent speech** is placed at the start of the prompt for immediate visual feedback.
- **Live Transcription**: High-performance audio-to-text using **OpenAI Whisper** (Medium model).
- **Multilingual Translation**: Automatically translates Chinese, Cantonese, Spanish, and other languages into English in real-time, allowing non-English speakers to control the visual engine seamlessly.
- **Voice Activity Detection (VAD)**: Smart volume gating and a 5-second auto-reset timer to prevent "ghost" transcriptions and hallucinations.
- **Token Management**: 12-second rolling buffer to ensure prompts stay within SDXL's 77-token limit.
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

The system utilizes a specialized template to maintain a consistent "Cultural Fusion" aesthetic while dynamically injecting speaker identity:

```text
A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age} {gender}, 
capturing a diverse Chinese-American identity, blending modern US urban settings with 
subtle traditional Chinese cultural motifs and textures, 8k UHD, highly detailed...
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
    Set your Capriole API key (if using cloud models) in your environment variables:
    ```powershell
    $env:CAPRIOLE_API_KEY = "your_key_here"
    ```

## 🕹️ Usage

1.  **Open TouchDesigner**: Load your StreamDiffusionTD project and ensure the OSC In DAT is listening on **Port 7000**.
2.  **Start the Pipeline**:
    ```bash
    python transcriber.py
    ```
3.  **Speak & Control**: The system will automatically capture your speech. Use the keys above to shift the identity of the generated figures as you talk.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

