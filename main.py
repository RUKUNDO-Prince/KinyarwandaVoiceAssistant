import json
import re
import os
import time
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from fuzzywuzzy import fuzz
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS
import torch
import webrtcvad  # Added for voice activity detection
import noisereduce as nr  # Added for noise reduction

# Create the necessary folders
os.makedirs("audio_inputs", exist_ok=True)
os.makedirs("audio_outputs", exist_ok=True)

# Load KinyaWhisper model & processor
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")


def transcribe_audio(audio_path):
    try:
        # Load and resample audio to 16kHz
        audio_data, _ = librosa.load(audio_path, sr=16000)

        # Convert to mono and normalize audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Reduce background noise
        audio_data = nr.reduce_noise(y=audio_data, sr=16000, stationary=True)

        # Normalize audio volume
        audio_data = librosa.util.normalize(audio_data) * 0.9

        # Remove silence using VAD with error handling
        try:
            vad = webrtcvad.Vad(1)  # Reduced aggressiveness to level 1
            frame_duration = 30  # ms
            frame_length = int(16000 * frame_duration / 1000)
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=frame_length)

            speech_frames = []
            for frame in frames.T:
                if vad.is_speech(frame.astype(np.int16).tobytes(), 16000):
                    speech_frames.append(frame)

            if len(speech_frames) == 0:
                print("No speech detected, using original audio")
                speech_frames = [audio_data]  # Fallback to original audio

            audio_data = np.concatenate(speech_frames)
        except Exception as vad_error:
            print(f"VAD processing error: {vad_error}, using original audio")
            pass  # Fallback to original audio if VAD fails

        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")

        # Generate transcription with anti-repetition parameters
        predicted_ids = model.generate(
            inputs.input_features,
            num_beams=5,
            repetition_penalty=1.5,
            temperature=0.8,
            max_length=448,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode and clean output
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Post-processing filters
        transcription = re.sub(r'(\w)\1{2,}', r'\1', transcription)
        transcription = re.sub(r'\b(\w+)( \1\b)+', r'\1', transcription)
        transcription = transcription.strip()

        return transcription
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

# Load Q&A pairs
with open("nlp_mapping.json") as f:
    qa_data = json.load(f)
    qa_pairs = {pair["question"]: pair["answer"] for pair in qa_data["qa_pairs"]}
    default_response = qa_data["default_response"]


def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def get_answer(question):
    normalized_question = normalize(question)
    best_score = 0
    best_answer = default_response
    matched_key = None

    for key in qa_pairs:
        score = fuzz.ratio(normalize(key), normalized_question)
        if score > best_score and score > 70:
            best_score = score
            best_answer = qa_pairs[key]
            matched_key = key

    return best_answer, matched_key


def process_audio(audio_path):
    timestamp = str(int(time.time()))
    input_path = f"audio_inputs/input_{timestamp}.wav"
    output_path = f"audio_outputs/output_{timestamp}.mp3"

    try:
        # Load and preprocess audio
        audio_data, _ = librosa.load(audio_path, sr=16000)
        sf.write(input_path, audio_data, 16000, subtype='PCM_16')

        question = transcribe_audio(input_path)
        answer, matched_key = get_answer(question)

        # Generate TTS response with proper language code
        tts = gTTS(text=answer, lang="en", slow=False)
        tts.save(output_path)

        return question, answer, matched_key, output_path
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return "Error during processing", default_response, None, None


def create_qa_reference():
    qa_list = ["**Supported Questions and Answers:**"]
    qa_list.append(f"Default Response: {default_response}\n")

    for idx, (question, answer) in enumerate(qa_pairs.items(), 1):
        qa_list.append(f"{idx}. **Question**: {question}")
        qa_list.append(f"   **Answer**: {answer}\n")

    return "\n".join(qa_list)


# Create Gradio interface
with gr.Blocks(title="Kinyarwanda Voice Assistant") as demo:
    gr.Markdown("# 🤖 Kinyarwanda Voice Assistant")
    gr.Markdown("Record or upload audio in Kinyarwanda to interact with the assistant")

    # QA Reference Section
    with gr.Accordion("📚 Click to see supported questions and answers", open=False):
        gr.Markdown(create_qa_reference())

    with gr.Row():
        audio_input = gr.Audio(
            label="Speak or Upload Audio",
            sources=["microphone", "upload"],
            type="filepath",
            format="wav"
        )
        output_audio = gr.Audio(label="Response Audio", autoplay=True)

    text_outputs = gr.Textbox(label="Conversation History", lines=4)

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")


    def process_and_display(audio_path):
        if not audio_path:
            raise gr.Error("Please provide an audio file first!")

        question, answer, matched_key, output_path = process_audio(audio_path)
        display_text = f"""
        🎤 Original Transcription: {question}
        🔍 Matched Question Key: {matched_key if matched_key else "No close match found"}
        🤖 Assistant Response: {answer}
        """
        return {
            output_audio: output_path if output_path else None,
            text_outputs: display_text
        }


    submit_btn.click(
        fn=process_and_display,
        inputs=audio_input,
        outputs=[output_audio, text_outputs]
    )

    clear_btn.click(
        fn=lambda: [None, None, ""],
        outputs=[audio_input, output_audio, text_outputs]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)