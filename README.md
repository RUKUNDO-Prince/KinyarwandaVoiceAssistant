# Kinyarwanda Voice Assistant 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

An intelligent voice assistant for Kinyarwanda language interaction, developed as part of the Intelligent Robotics course.

![Interface Demo](media/interface.png)  

## Features 🌟
- 🎙️ **Kinyarwanda ASR** using KinyaWhisper (16kHz optimized)
- 🧠 **Contextual Understanding** with fuzzy logic matching
- 📢 **Natural Responses** with Kinyarwanda TTS
- 🔇 **Noise Reduction** using advanced audio cleaning
- 🎚️ **Voice Activity Detection** for precise speech recognition
- 🔄 **Anti-Repetition** transcription filters
- 📊 **Conversation Analytics** with matching insights
- 🌐 **Web Interface** with Gradio integration

## Tech Stack 🛠️
- **Core AI**: Hugging Face Transformers
- **Audio Processing**: Librosa + Soundfile
- **NLP**: FuzzyWuzzy + Python-Levenshtein
- **Interface**: Gradio
- **Optimization**: WebRTC VAD + Noisereduce

## Installation 💻

### Prerequisites
- Python 3.12
- FFmpeg (audio processing):
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Windows (via chocolatey)
  choco install ffmpeg
  ```
  
## Quick Start 🚀

- Clone repository
  ```bash
    git clone https://github.com/RUKUNDO-Prince/KinyarwandaVoiceAssistant.git
    cd KinyarwandaVoiceAssistant
  ```
- Set up virtual environment
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/macOS
  .\.venv\Scripts\activate   # Windows

  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## Configuration ⚙️

#### QA Configuration in `nlp_mapping.json`

  ```json
  {
    "qa_pairs": [
      {
        "question": "Mwaramuce neza?",
        "answer": "Mwaramutse! Amakuru yanyu?"
      }
    ],
    "default_response": "Vugurura ikibazo."
  }
  ```

#### Audio Files
You can find sample Kinyarwanda recordings in the `/sample_inputs` folder

Supported formats: `WAV`, `MP3`, `OGG`


## Usage 🚀

#### Start the application
  ```bash
  python main.py
  ```

#### Access the interface
- Navigate to http://localhost:7860

## Interface Guide 💡
1. Record using your microphone or upload an audio file
1. Click **Submit** to process (⏳ ~10–60 sec)
1. Response audio auto-plays
1. Review **Conversation History**:
   - Raw transcription
   - Matched question key
   - Generated response
1. Click **Clear** to reset session

## Example Interactions 🗣️
| Raw Transcription | Matched Question | System Response                                              |
|-------------------|-----------|--------------------------------------------------------------|
| mizeneza          |Umeze neza?| Yego! Turashima Imana.                                       |
| wakorewee heheh   |Wakorewe hehe?| Nakorewe muri Rwanda Coding Academy, nakozwe na Remy Chiesa. |