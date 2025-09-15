import os
import logging
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
import streamlit as st

# Optional imports
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from transformers import pipeline
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
    # ✅ Fix: use a variable key name, not the actual key string
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
def generate_response(user_input: str) -> str:
    label = classify_text(user_input)
    logger.info(f"Classifier prediction: {label}")

    # Strong Gemini prompt for unique answers
    prompt = f"""
    You are a warm, empathetic mental health companion.
    The user said: "{user_input}"
    
    Based on this, provide 3–5 practical, supportive, and fresh coping tips.
    Avoid repeating the same tips word-for-word every time.
    Use simple language, keep answers under 120 words.
    """

    # Try Gemini first
    if GEMINI_MODEL:
        try:
            resp = GEMINI_MODEL.generate_content(prompt)
            if resp and getattr(resp, "text", None):
                return resp.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")

    # Fallback guides with randomization
    guides = {
        "stress": [
            "😌 Try deep breathing for 2 minutes",
            "💧 Stay hydrated and take a short walk",
            "📝 Write your worries down before sleeping"
        ],
        "panic": [
            "😮‍💨 Focus on 5-4-3-2-1 grounding",
            "📱 Call a friend you trust immediately",
            "🎧 Listen to calming sounds"
        ],
        "depression": [
            "🌧 Create a small daily goal (like making your bed)",
            "💚 Text one person today, even a simple 'hi'",
            "📖 Journal one positive thing about your day"
        ],
        "sleep": [
            "🌙 Keep lights dim before bed",
            "📵 No screens 1 hour before sleep",
            "🛏️ Try a warm shower before bed"
        ]
    }

    if label in guides:
        import random
        return random.choice(guides[label]) + "\n\n📞 Crisis Resources: 988 or Text HOME to 741741"

    # If everything else fails → GPT-2
    if generator:
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






