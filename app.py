import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load model and vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    tfidf, model = pickle.load(f)

# Stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write(
    "Enter a news article below to check whether it is **FAKE** or **REAL**.\n\n"
    "_For best results, paste the full article text._"
)

user_input = st.text_area("Paste the news text here:", height=250)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])

        # Prediction probabilities
        proba = model.predict_proba(vectorized_text)[0]

        fake_index = list(model.classes_).index("FAKE")
        real_index = list(model.classes_).index("REAL")

        fake_prob = proba[fake_index]
        real_prob = proba[real_index]

        st.write(f"Confidence → REAL: {real_prob:.2f} | FAKE: {fake_prob:.2f}")

        # Decision logic
        if real_prob > 0.55:
            st.success("✅ This news is likely REAL")
        elif fake_prob > 0.55:
            st.error("🚨 This news is likely FAKE")
        else:
            st.warning("⚠️ The model is unsure about this news")
