import os
import streamlit as st
from transformers import pipeline
from datetime import datetime
import logging
import pandas as pd

# --------------------------------------------------------
# Logging setup
# --------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# --------------------------------------------------------
# Hugging Face cache setup
# --------------------------------------------------------
cache_dir = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

try:
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Created cache directory: {cache_dir}")
except Exception as e:
    logger.error(f"Cache directory error: {e}")
    st.error(f"Cannot create cache directory: {e}. Please try refreshing or contact support.")
    st.stop()

# --------------------------------------------------------
# Load model (use gpt2-medium for fallback responses)
# --------------------------------------------------------
try:
    generator = pipeline('text-generation', model='gpt2-medium', cache_dir=cache_dir)
    logger.info("Successfully loaded gpt2-medium model")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    st.error(f"Error loading model: {e}. Please try refreshing or contact support.")
    st.stop()

# --------------------------------------------------------
# Topic-specific guides
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
📞 *Crisis Resources (If You’re Struggling)*

- 988 Suicide & Crisis Lifeline: Call/text 988 (24/7, free, confidential).  
- Crisis Text Line: Text HOME to 741741 (free, anonymous, 24/7).  
- Soluna App: Free for ages 13–25 (iOS/Android).  

⚠ Reminder: This app is not medical advice—please consult a licensed professional if you’re in crisis.
"""

# --------------------------------------------------------
# AI Response generator with topic-specific guides
# --------------------------------------------------------
def generate_response(user_input):
    text = user_input.lower()
text_clean = text.translate(str.maketrans('', '', string.punctuation))

    greetings = ["hi", "hello", "hey", "how are you", "how are you doing", "good morning", "good afternoon", "good evening"]

    # check if text_clean is exactly one of greetings or starts with one
    if any(text_clean == greet or text_clean.startswith(greet + " ") for greet in greetings):
        return "Hello! 😊 How can I support you today?"

        
    if any(word in text for word in ["panic", "anxiety", "attack"]):
        return panic_guide + "\n\n" + resources
    elif any(word in text for word in ["stress", "stressed", "pressure"]):
        return stress_guide
    elif any(word in text for word in ["depressed", "sad", "low", "hopeless"]):
        return depression_guide + "\n\n" + resources
    elif any(word in text for word in ["sleep", "insomnia", "tired"]):
        return sleep_guide
    else:
        # fallback: AI-generated advice
        prompt = (
            f"You are a supportive AI wellness guide. The user said: '{user_input}'.\n"
            f"Give practical advice in 3–5 clear steps. Keep it warm and useful."
        )
        try:
            response = generator(
                prompt,
                max_length=250,
                num_return_sequences=1,
                temperature=0.85,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=generator.tokenizer.eos_token_id
            )[0]['generated_text']
            return response.replace(prompt, "").strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I couldn’t generate advice right now. Try again."

# --------------------------------------------------------
# Streamlit App
# --------------------------------------------------------
st.title("🧠 Mental Health Helper")
st.write("A safe space to get advice, therapy tips, panic attack help, and track your mood. Not a replacement for therapy.")

# --------------------------
# Feature 1: Chatbot (always fresh)
# --------------------------
st.subheader("💬 Chat for Advice")
user_input = st.text_input("What’s on your mind? (e.g., 'I'm stressed'):")

if st.button("Get Advice") and user_input:
    # clear previous chat automatically
    st.session_state['messages'] = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    ai_response = generate_response(user_input)
    st.session_state.messages.append({"role": "ai", "content": ai_response})

    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "AI"
        st.markdown(f"{role}:** {message['content']}")

# --------------------------
# Feature 2: Therapy Tips (general, always available)
# --------------------------
st.subheader("🌱 General Therapy & Self-Help Options")
with st.expander("Click to view general therapy practices"):
    st.write("""
    - 📝 Journaling: Helps track mood triggers and progress.  
    - 🧘 Mindfulness: Try 5–10 minutes daily meditation or mindful walking.  
    - 🎶 Behavioral Activation: Schedule enjoyable small activities daily.  
    - 👥 Group Therapy: Talking with others reduces isolation.  
    """)

# --------------------------
# Feature 3: Mood Tracking (fresh each time)
# --------------------------
st.subheader("📊 Track Your Mood")
mood = st.slider("How’s your mood today? (1 = low, 5 = high)", 1, 5, 3)

if st.button("Log Mood"):
    # clear old moods each time
    st.session_state['moods'] = []
    st.session_state.moods.append({
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'mood': mood
    })

if 'moods' in st.session_state and st.session_state.moods:
    st.write("Your Mood (Latest Log Only):")
    df = pd.DataFrame(st.session_state.moods)
    df['date'] = pd.to_datetime(df['date'])
    st.line_chart(df.set_index('date')['mood'])

st.write("Prototype v11.0: Topic-specific solutions with auto-clearing history.")
