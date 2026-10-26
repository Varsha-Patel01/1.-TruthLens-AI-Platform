
import joblib

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

title = "NASA Perseverance Rover Explores Mars"

text = """
NASA Perseverance rover landed on Mars on February 18, 2021.
The rover is exploring Jezero Crater, studying the planet's
geology and climate, searching for signs of ancient microbial
life, and collecting rock and soil samples.
"""

tests = {
    "Title only": title,
    "Article text only": text,
    "Title + article": title + " " + text
}

for name, content in tests.items():
    features = vectorizer.transform([content])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print("\nTest:", name)
    print("Prediction:", "REAL NEWS" if prediction == 1 else "FAKE NEWS")
    print(f"Fake probability: {probabilities[0] * 100:.2f}%")
    print(f"Real probability: {probabilities[1] * 100:.2f}%")
 