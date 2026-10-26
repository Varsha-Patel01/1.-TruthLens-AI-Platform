
import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("=" * 60)
print("TRUTHLENS AI - MODEL TRAINING")
print("=" * 60)

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)

print("\nOriginal dataset size:", len(data))

# =========================================================
# 2. CLEAN TEXT
# =========================================================

data = data.dropna(subset=["title", "text"]).copy()

data["title"] = data["title"].astype(str).str.strip()
data["text"] = data["text"].astype(str).str.strip()

data["content"] = data["title"] + " " + data["text"]

# Normalize text for duplicate detection
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["normalized_content"] = data["content"].apply(normalize_text)

# Remove articles that have contradictory labels
label_counts = data.groupby("normalized_content")["label"].nunique()
conflicting_articles = label_counts[label_counts > 1].index

print("Conflicting articles removed:", len(conflicting_articles))

data = data[
    ~data["normalized_content"].isin(conflicting_articles)
].copy()

# Remove duplicate articles
data = data.drop_duplicates(
    subset=["normalized_content"]
).copy()

print("Dataset size after cleaning:", len(data))
print("\nClass distribution:")
print(data["label"].value_counts().rename(
    index={0: "Fake", 1: "Real"}
))

# =========================================================
# 3. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    data["content"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =========================================================
# 4. TF-IDF FEATURE EXTRACTION
# =========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")

# =========================================================
# 5. TRAIN MODEL
# =========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1500,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Training completed!")

# =========================================================
# 6. EVALUATE MODEL
# =========================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

# =========================================================
# 7. SAVE MODEL AND VECTORIZER
# =========================================================

os.makedirs("model", exist_ok=True)

print("\nSaving model files...")

with open("model/fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel saved: model/fake_news_model.pkl")
print("Vectorizer saved: model/tfidf_vectorizer.pkl")

# =========================================================
# 8. TEST AN UNSEEN NASA EXAMPLE
# =========================================================

print("\n" + "=" * 60)
print("NASA ARTICLE CHECK - EXTRA TEST")
print("=" * 60)

nasa_title = "NASA Perseverance Rover Explores Mars"

nasa_text = """
NASA Perseverance rover landed on Mars on February 18, 2021.
The rover is exploring Jezero Crater, studying the planet's
geology and climate, searching for signs of ancient microbial
life, and collecting rock and soil samples.
"""

nasa_features = vectorizer.transform([
    nasa_title + " " + nasa_text
])

nasa_prediction = model.predict(nasa_features)[0]
nasa_probabilities = model.predict_proba(nasa_features)[0]

print(
    "Prediction:",
    "REAL NEWS" if nasa_prediction == 1 else "FAKE NEWS"
)
print(f"Fake probability: {nasa_probabilities[0] * 100:.2f}%")
print(f"Real probability: {nasa_probabilities[1] * 100:.2f}%")

print("\nTraining and evaluation finished!")
print("Note: NASA result is a separate example, not part of test accuracy.")
 