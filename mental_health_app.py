import os
import logging
import pandas as pd
from datetime import datetime
import threading
from flask import Flask, request, jsonify
import streamlit as st

# Optional imports
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from transformers import pipeline, pipeline as hf_pipeline
except Exception:
    pipeline = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------
# Logging
# --------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Curated guides
# --------------------------------------------------------
stress_guide = "😌 Coping with Stress...\n1. Take short breaks...\n2. Deep breathing..."
panic_guide = "😮‍💨 Panic/Anxiety Help...\n1. Remind yourself it will pass..."
depression_guide = "🌧 Feeling Low/Depressed...\n1. Keep routine...\n2. Schedule enjoyable activities..."
sleep_guide = "🌙 Better Sleep Tips...\n1. Fixed bedtime...\n2. Avoid caffeine before sleep..."

resources = """
📞 Crisis Resources:
- 988 Suicide & Crisis Lifeline
- Crisis Text Line: HOME to 741741
⚠ Not a replacement for professional help.
"""

# --------------------------------------------------------
# Custom Classifier (Tiny AI Model)
# --------------------------------------------------------
categories = ["stress", "panic", "depression", "sleep", "general"]

train_data = [
    ("I am stressed with exams", "stress"),
    ("Too much pressure at work", "stress"),
    ("I am panicking", "panic"),
    ("Having an anxiety attack", "panic"),
    ("I feel very sad and hopeless", "depression"),
    ("I think I’m depressed", "depression"),
    ("I can’t sleep at night", "sleep"),
    ("Suffering from insomnia", "sleep"),
    ("Just need some advice", "general"),
    ("What can I do to feel better?", "general")
]

X_train, y_train = zip(*train_data)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)

classifier = LogisticRegression()
classifier.fit(X_vec, y_train)

def classify_text(user_input: str) -> str:
    X_test = vectorizer.transform([user_input])
    pred = classifier.predict(X_test)[0]
    return pred

# --------------------------------------------------------
# Hugging Face fallback
# --------------------------------------------------------
generator = None
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

if pipeline:
    try:
        generator = pipeline("text-generation", model="gpt2-medium", cache_dir=cache_dir)
        logger.info("HF GPT-2 fallback ready")
    except Exception as e:
        logger.error(f"HF model load failed: {e}")

# --------------------------------------------------------
# Google Gemini Setup
# --------------------------------------------------------
GEMINI_MODEL = None
if genai is not None:
    key = st.secrets.get("AIzaSyAttoi7RF50jBTnBYHSqpgIbKzPsRx0ZME", os.getenv("AIzaSyAttoi7RF50jBTnBYHSqpgIbKzPsRx0ZME"))
    if key:
        try:
            genai.configure(api_key=key)
            GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini model initialized ✅")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

# --------------------------------------------------------
# Hybrid Response Generator
# --------------------------------------------------------
def expand_with_gemini(base_text: str) -> str:
    if GEMINI_MODEL:
        try:
            resp = GEMINI_MODEL.generate_content(
                f"Expand this wellness guide with 2–3 empathetic tips:\n\n{base_text}"
            )
            if resp and getattr(resp, "text", None):
                return base_text + "\n\n💡 Extra AI Tips:\n" + resp.text.strip()
        except Exception as e:
            logger.error(f"Gemini expand failed: {e}")
    return base_text

def generate_response(user_input: str) -> str:
    label = classify_text(user_input)
    logger.info(f"Classifier prediction: {label}")

    if label == "panic":
        return expand_with_gemini(panic_guide + "\n\n" + resources)
    elif label == "stress":
        return expand_with_gemini(stress_guide)
    elif label == "depression":
        return expand_with_gemini(depression_guide + "\n\n" + resources)
    elif label == "sleep":
        return expand_with_gemini(sleep_guide)
    else:  # general → send directly to Gemini
        if GEMINI_MODEL:
            try:
                prompt = f"You are a supportive assistant. User: {user_input}\nGive 3–5 clear, practical, empathetic tips."
                resp = GEMINI_MODEL.generate_content(prompt)
                if resp and getattr(resp, "text", None):
                    return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini general failed: {e}")

        if generator:  # fallback GPT-2
            try:
                out = generator(user_input, max_length=200, num_return_sequences=1)
                return out[0]["generated_text"]
            except Exception as e:
                logger.error(f"HF fallback failed: {e}")

        return "Sorry, I couldn’t generate advice right now."

# --------------------------------------------------------
# Flask API
# --------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/api/respond", methods=["POST"])
def respond():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    reply = generate_response(text)
    return jsonify({"response": reply})




