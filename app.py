import streamlit as st
import os
import tempfile
import speech_recognition as sr
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from moviepy.editor import VideoFileClip
from pytesseract import image_to_string
from PIL import Image
import cv2
from transformers import pipeline
import concurrent.futures

# Downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Use faster summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Page config
st.set_page_config(page_title="🎓 Smart Lecture Notes", layout="centered", page_icon="🎙️")

st.title("🎓 Smart Lecture Notes")
st.caption("Powered by AI for transcription, summarization, OCR, and more.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    1. Upload an audio or video lecture (any format).
    2. The app transcribes speech and extracts slide/board text with OCR.
    3. Then it summarizes the content and extracts keywords.
    """)

uploaded_file = st.file_uploader("📤 Upload Lecture File (Audio/Video - Any Format)", type=None)

# Functions
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source, duration=30)  # limit to 30s
    return recognizer.recognize_google(audio)

def extract_keywords(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    stemmer = PorterStemmer()
    return list(set(f"{stemmer.stem(w.lower())} ({t})" for w, t in pos_tags if t.startswith("NN") or t.startswith("VB")))

def summarize_text(text, ratio="short"):
    max_len, min_len = (100, 30) if ratio == "short" else (150, 50) if ratio == "medium" else (250, 80)
    if len(text.split()) < min_len:
        return "Transcript is too short to summarize."
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        sum_out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summary += sum_out[0]['summary_text'] + " "
    return summary.strip()

def extract_slide_text(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    ocr_texts = set()
    while cap.isOpened() and frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = image_to_string(image)
            if text.strip():
                ocr_texts.add(text.strip())
        frame_count += 1
    cap.release()
    return "\n\n".join(ocr_texts)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    audio_path = file_path
    slide_text = ""

    try:
        if file_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            audio_path = extract_audio(file_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            st.info("🚀 Running OCR and transcription in parallel...")
            ocr_future = executor.submit(extract_slide_text, file_path) if file_path.endswith((".mp4", ".mov", ".avi", ".mkv")) else None
            trans_future = executor.submit(transcribe_audio, audio_path)

            transcript = trans_future.result()
            slide_text = ocr_future.result() if ocr_future else ""

        if slide_text:
            st.subheader("🖼️ Slide/Whiteboard Extracted Text")
            st.code(slide_text, language="text")

        st.subheader("📜 Full Transcription")
        st.text_area("Transcript", transcript, height=200)

        st.subheader("🔑 Extracted Keywords")
        keywords = extract_keywords(transcript)
        st.success(", ".join(keywords))

        st.subheader("📋 Lecture Summary")
        if len(transcript.split()) < 30:
            st.warning("Transcript too short for a meaningful summary.")
        else:
            summary_mode = st.selectbox("📝 Choose Summary Detail", ["short", "medium", "detailed"])
            summary = summarize_text(transcript, ratio=summary_mode)
            st.text_area("Summary", summary, height=200)

        st.success("✅ Lecture notes generated successfully!")

    except Exception as e:
        st.error(f"🚫 Error: {e}")

    finally:
        os.remove(file_path)
        if audio_path != file_path and os.path.exists(audio_path):
            os.remove(audio_path)
