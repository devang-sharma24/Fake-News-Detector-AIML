import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("data/news.csv")

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
df["clean_content"] = df["content"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_content"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Final Model Accuracy:", accuracy)

# -------- SAVE MODEL --------
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((tfidf, model), f)

print("Model saved as model/fake_news_model.pkl")
