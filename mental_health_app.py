import os
import streamlit as st
import logging
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from transformers import pipeline
import threading
from flask import Flask, request, jsonify

# --------------------------------------------------------
# Logging setup
# --------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Hugging Face fallback setup (gpt2-medium)
# --------------------------------------------------------
cache_dir = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception as e:
    st.error(f"Cache directory error: {e}")
    st.stop()

try:
    generator = pipeline('text-generation', model='gpt2-medium', cache_dir=cache_dir)
    logger.info("Loaded gpt2-medium model")
except Exception as e:
    st.error(f"Error loading GPT-2 model: {e}")
    generator = None

# --------------------------------------------------------
# Configure Gemini (Google AI Studio)
# --------------------------------------------------------
GEMINI_MODEL = None
if "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini model ready")
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")

# --------------------------------------------------------
# Guides
# --------------------------------------------------------

stress_guide = """
😌 *Coping with Stress*

1. Take short breaks every 1–2 hours of work/study.  
2. Practice deep breathing (inhale 4s, exhale 6s).  
3. Break big tasks into smaller ones.  
4. Write down your worries, then list 1 small action for each.  
5. Try light exercise (walk, stretch, yoga).  
"""

panic_guide = """
😮‍💨 *Panic/Anxiety Attack Help*

1. Remind yourself: "This will pass."  
2. Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s).  
3. Use grounding: 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste.  
4. Splash cold water on your face.  
"""



depression_guide = """
🌧 *When Feeling Low/Depressed*

1. Keep a daily routine (wake/sleep times).  
2. Schedule at least one enjoyable activity daily.  
3. Reach out to a friend or family member, even briefly.  
4. Avoid isolation—spend time in sunlight/nature.  
5. Journal your feelings, note small positives each day.  
"""



sleep_guide = """
🌙 *Better Sleep Tips*

1. Keep a fixed bedtime and wake-up time.  
2. Avoid caffeine or phone screens 2 hours before bed.  
3. Use relaxation routines: soft music, reading, or meditation.  
4. Keep your room dark, cool, and quiet.  

"""



resources = """
📞 *Crisis Resources*

- 988 Suicide & Crisis Lifeline: Call/text 988 (24/7, free, confidential).  
- Crisis Text Line: Text HOME to 741741 (free, anonymous, 24/7).  
- Soluna App: Free for ages 13–25 (iOS/Android).  

⚠ Reminder: This app is not medical advice—please consult a licensed professional if you’re in crisis.

"""
# --------------------------------------------------------
# Response generator (Hybrid: guide + Gemini expansion)
# --------------------------------------------------------
def expand_with_gemini(base_text: str) -> str:
    if GEMINI_MODEL:
        try:
            response = GEMINI_MODEL.generate_content(
                f"Expand this wellness guide with 2–3 additional practical, empathetic tips:\n\n{base_text}"
            )
            if response and response.text:
                return base_text + "\n\n💡 Extra AI Tips:\n" + response.text.strip()
        except Exception as e:
            logger.error(f"Gemini expansion failed: {e}")
    return base_text

def generate_response(user_input: str) -> str:
    text = user_input.lower()

    if any(word in text for word in ["panic", "anxiety", "attack"]):
        return expand_with_gemini(panic_guide + "\n\n" + resources)
    elif any(word in text for word in ["stress", "stressed", "pressure"]):
        return expand_with_gemini(stress_guide)
    elif any(word in text for word in ["depressed", "sad", "low", "hopeless"]):
        return expand_with_gemini(depression_guide + "\n\n" + resources)
    elif any(word in text for word in ["sleep", "insomnia", "tired"]):
        return expand_with_gemini(sleep_guide)
    else:
        prompt = (
            f"You are a supportive mental health assistant.\n"
            f"User: '{user_input}'\n"
            f"Give 3–5 clear, practical, empathetic tips."
        )
        if GEMINI_MODEL:
            try:
                response = GEMINI_MODEL.generate_content(prompt)
                return response.text.strip() if response and response.text else "Couldn’t generate advice."
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        if generator:
            try:
                result = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
                return result.replace(prompt, "").strip()
            except Exception as e:
                logger.error(f"GPT-2 error: {e}")
        return "Sorry, I couldn’t generate advice right now."

# --------------------------------------------------------
# Flask API for frontend
# --------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/api/respond", methods=["POST"])
def respond():
    data = request.get_json()
    user_text = data.get("text", "")
    response_text = generate_response(user_text)
    return jsonify({"response": response_text})

def run_flask():
    flask_app.run(port=8502)  # separate port from Streamlit

threading.Thread(target=run_flask, daemon=True).start()

# --------------------------------------------------------
# Streamlit UI (still works!)
# --------------------------------------------------------
st.title("🧠 Mental Health Helper")
st.write("A safe space for advice, therapy tips, panic attack help, and mood tracking. *Not a replacement for therapy.*")

st.subheader("💬 Chat for Advice")
user_input = st.text_input("What’s on your mind? (e.g., 'I'm stressed')")

if st.button("Get Advice") and user_input:
    if "messages" not in st.session_state:
        st.session_state['messages'] = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    ai_response = generate_response(user_input)
    st.session_state.messages.append({"role": "ai", "content": ai_response})

if "messages" in st.session_state:
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "AI"
        with st.chat_message(role.lower()):
            st.markdown(message["content"])

st.subheader("🌱 General Therapy & Self-Help")
with st.expander("Click to view practices"):
    st.write("""Cognitive Behavioral Techniques (CBT)

Thought Challenging: Identify negative automatic thoughts and question their accuracy.
Behavioral Activation: Plan small, rewarding activities daily to improve motivation and mood.
Problem-Solving: Break overwhelming challenges into smaller, actionable steps.
Exposure Practice: Gradually face avoided situations to reduce fear and anxiety.

 Mindfulness & Relaxation

Progressive Muscle Relaxation (PMR): Tense and release muscle groups to ease physical tension.
Mindful Breathing: Focus on slow, deep breathing for 5–10 minutes daily.
Grounding Techniques: Use the 5-4-3-2-1 sensory method to stay present during stress.
Body Scan Meditation: Notice sensations in each part of your body to calm the mind.

💚 Self-Care Strategies

Sleep Hygiene: Stick to a consistent bedtime and wake-up time.
Physical Activity: Regular walking, stretching, or exercise to boost mood.
Social Connection: Maintain supportive relationships and talk about your feelings.
Journaling: Track thoughts, emotions, and patterns to increase self-awareness.

🆘 Crisis Management

Safety Planning: Identify warning signs, coping tools, and safe contacts.
Crisis Lines: Know local/national emergency resources (like 988 in the US, 14416 in India).
Emergency Support Network: Share a list of trusted contacts you can reach out to.
Professional Help: Seek immediate support if you’re at risk of harming yourself.""")

st.subheader("📊 Track Your Mood")
mood = st.slider("How’s your mood today? (1 = low, 5 = high)", 1, 5, 3)

if st.button("Log Mood"):
    if "moods" not in st.session_state:
        st.session_state['moods'] = []
    st.session_state.moods.append({
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'mood': mood
    })

if "moods" in st.session_state and st.session_state.moods:
    df = pd.DataFrame(st.session_state.moods)
    df['date'] = pd.to_datetime(df['date'])
    st.line_chart(df.set_index('date')['mood'])

st.caption("Prototype v12.1: Now with API route for frontend integration.")


