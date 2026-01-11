import pandas as pd

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Standardize labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Keep required columns only
columns = ["title", "text", "subject", "date", "label"]
fake_df = fake_df[columns]
true_df = true_df[columns]

# Merge datasets
news_df = pd.concat([fake_df, true_df], ignore_index=True)

# Remove rows with missing text or title
news_df = news_df.dropna(subset=["title", "text"])

# Save final dataset
news_df.to_csv("data/news.csv", index=False, encoding="utf-8")

print("news.csv created successfully!")
print("Shape:", news_df.shape)
