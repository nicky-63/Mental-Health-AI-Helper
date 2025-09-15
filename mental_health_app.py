import os
import streamlit as st
import logging
import pandas as pd
from datetime import datetime
import threading
from flask import Flask, request, jsonify

# Optional imports (wrap in try/except so app still runs if these libs aren't available)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

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
    logger.info(f"HF cache dir ensured at {cache_dir}")
except Exception as e:
    logger.error(f"Could not create cache dir: {e}")

generator = None
if pipeline:
    try:
        # Using gpt2-medium as fallback (you can replace with distilgpt2 if you want faster starts)
        generator = pipeline("text-generation", model="gpt2-medium", cache_dir=cache_dir)
        logger.info("Loaded gpt2-medium fallback model")
    except Exception as e:
        logger.error(f"Error loading HF model: {e}")
        generator = None
else:
    logger.warning("transformers.pipeline not available; GPT-2 fallback disabled.")

# --------------------------------------------------------
# Configure Gemini (Google AI Studio)
# --------------------------------------------------------
GEMINI_MODEL = None
if genai is not None:
    google_key = None
    # Prefer Streamlit secrets, fallback to env var
    if "GOOGLE_API_KEY" in st.secrets:
        google_key = st.secrets["GOOGLE_API_KEY"]
    elif os.getenv("GOOGLE_API_KEY"):
        google_key = os.getenv("GOOGLE_API_KEY")

    if google_key:
        try:
            genai.configure(api_key=google_key)
            # NOTE: API surface may differ — keep this guard so the app won't crash if names differ
            try:
                GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini model initialized")
            except Exception as e:
                # if the library offers a different constructor, still continue gracefully
                logger.info("Gemini library present but GenerativeModel(...) failed — continuing without Gemini.")
                GEMINI_MODEL = None
        except Exception as e:
            logger.error(f"Failed to configure genai: {e}")
            GEMINI_MODEL = None
    else:
        logger.info("No Google API key found; Gemini disabled.")
else:
    logger.info("google.generativeai not installed; Gemini integration disabled.")

# --------------------------------------------------------
# Helpful guides (static)
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
# AI helper functions (Gemini + HF fallback)
# --------------------------------------------------------
def expand_with_gemini(base_text: str) -> str:
    """
    If Gemini is available, try to expand the base_text with a short list of extra tips.
    If not, return base_text unchanged.
    """
    if GEMINI_MODEL is None:
        return base_text

    try:
        # If your genai library uses a different method, adjust accordingly.
        # This is a best-effort call that may need to be adapted to the exact genai API you have.
        response = GEMINI_MODEL.generate_content(
            f"Expand this wellness guide with 2–3 additional practical, empathetic tips:\n\n{base_text}"
        )
        # response.text is the common field — guard access
        if response and getattr(response, "text", None):
            return base_text + "\n\n💡 Extra AI Tips:\n" + response.text.strip()
    except Exception as e:
        logger.error(f"Gemini expansion failed: {e}")

    return base_text


def generate_response(user_input: str) -> str:
    """
    Priority:
      1. Topic-specific static guide (and optionally expanded by Gemini)
      2. Gemini general response (if available)
      3. HF fallback text-generation (if available)
      4. Return a polite failure message
    """
    text = (user_input or "").lower()

    if any(word in text for word in ["panic", "anxiety", "attack"]):
        return expand_with_gemini(panic_guide + "\n\n" + resources)
    if any(word in text for word in ["stress", "stressed", "pressure"]):
        return expand_with_gemini(stress_guide)
    if any(word in text for word in ["depressed", "sad", "low", "hopeless"]):
        return expand_with_gemini(depression_guide + "\n\n" + resources)
    if any(word in text for word in ["sleep", "insomnia", "tired"]):
        return expand_with_gemini(sleep_guide)

    prompt = (
        f"You are a supportive mental health assistant.\n"
        f"User: '{user_input}'\n"
        f"Give 3–5 clear, practical, empathetic tips."
    )

    # Try Gemini for open-ended prompts
    if GEMINI_MODEL is not None:
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            if response and getattr(response, "text", None):
                return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generate error: {e}")

    # Fallback to HF generator
    if generator is not None:
        try:
            out = generator(prompt, max_length=200, num_return_sequences=1)
            if isinstance(out, list) and out:
                text_out = out[0].get("generated_text", "")
                # Remove the prompt prefix (if present)
                return text_out.replace(prompt, "").strip()
        except Exception as e:
            logger.error(f"HuggingFace generator error: {e}")

    return "Sorry, I couldn’t generate advice right now. Please try again later."

# --------------------------------------------------------
# Flask API for external integration
# --------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/api/respond", methods=["POST"])
def respond():
    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "")
    response_text = generate_response(user_text)
    return jsonify({"response": response_text})

def run_flask():
    # run with no reloader and pinned host so we don't create duplicate threads during Streamlit reloads
    try:
        flask_app.run(host="0.0.0.0", port=8502, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask failed to start: {e}")

# Start Flask only once per Streamlit session (avoids double threads on reruns)
if "flask_started" not in st.session_state:
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    st.session_state["flask_started"] = True
    logger.info("Started Flask thread on port 8502")

# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.set_page_config(page_title="Mental Health Helper", layout="centered")
st.title("🧠 Mental Health Helper")
st.write("A safe space for advice, therapy tips, panic attack help, and mood tracking. *Not a replacement for therapy.*")

# Initialize persistent session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "moods" not in st.session_state:
    st.session_state["moods"] = []

# Controls: reset button
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Reset History"):
        st.session_state["messages"] = []
        st.session_state["moods"] = []
        st.success("Chat and mood history cleared.")

with col2:
    st.caption("Tip: Use 'Reset History' before demos if you want a clean slate.")

# Chat UI
st.subheader("💬 Chat for Advice")
user_input = st.text_input("What’s on your mind? (e.g., 'I'm stressed')", key="user_input")

if st.button("Get Advice") and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    ai_resp = generate_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": ai_resp})
    # clear the text input widget after sending (works by resetting the key in session_state)
    st.session_state["user_input"] = ""

# Render chat messages using Streamlit chat UI
if st.session_state["messages"]:
    for message in st.session_state["messages"]:
        msg_role = message.get("role", "assistant")
        # Map roles to Streamlit chat names: 'user' or 'assistant'
        role_for_st = "user" if msg_role.lower() == "user" else "assistant"
        with st.chat_message(role_for_st):
            # Use markdown to keep formatting and newlines readable
            st.markdown(message.get("content", ""))

# Therapy resources / expander (static)
st.subheader("🌱 General Therapy & Self-Help")
with st.expander("Click to view practices"):
    st.write("""
**Cognitive Behavioral Techniques (CBT)**

- Thought Challenging: Identify negative automatic thoughts and question their accuracy.
- Behavioral Activation: Plan small, rewarding activities daily to improve motivation and mood.
- Problem-Solving: Break overwhelming challenges into smaller, actionable steps.
- Exposure Practice: Gradually face avoided situations to reduce fear and anxiety.

**Mindfulness & Relaxation**

- Progressive Muscle Relaxation (PMR): Tense and release muscle groups to ease physical tension.
- Mindful Breathing: Focus on slow, deep breathing for 5–10 minutes daily.
- Grounding Techniques: Use the 5-4-3-2-1 sensory method to stay present during stress.

**Self-Care & Crisis Management**

- Sleep Hygiene: Stick to a consistent bedtime and wake-up time.
- Social Connection: Maintain supportive relationships and talk about your feelings.
- Safety Planning & Crisis Lines: Know local/national emergency resources and create a safety plan.
""")

# Mood tracking
st.subheader("📊 Track Your Mood")
mood = st.slider("How’s your mood today? (1 = low, 5 = high)", 1, 5, 3, key="mood_slider")

if st.button("Log Mood"):
    st.session_state["moods"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mood": int(st.session_state.get("mood_slider", 3))
    })
    st.success("Mood logged.")

if st.session_state["moods"]:
    df = pd.DataFrame(st.session_state["moods"])
    df["date"] = pd.to_datetime(df["date"])
    st.write("Your Mood Logs:")
    st.line_chart(df.set_index("date")["mood"])

st.caption("Prototype v12.2: Persistent chat + mood logs, Flask API on port 8502 for integration.")



