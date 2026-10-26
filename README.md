# 📰 Fake News Detection Platform

An AI/ML based web application that predicts whether a given news
headline or article is likely to be **Fake** or **Real**.

## 🚀 Features

- 🔍 Fake vs Real news classification
- 📰 Headline analysis
- 📄 Full article analysis
- 📊 Confidence score
- 📈 Model performance dashboard
- 🕒 Prediction history
- 🌐 Streamlit web interface

## 🧠 Machine Learning

The project uses:

- **TF-IDF** for text feature extraction
- **Logistic Regression** for classification
- **Scikit-learn** for machine learning
- **Pandas** for data processing
- **Streamlit** for the web application

## 📊 Dataset

The dataset contains:

- Total records: **44,898**
- Fake news: **23,481**
- Real news: **21,417**

The dataset contains news:

- Title
- Article Text
- Subject
- Date
- Label

## 📈 Model Performance

The Logistic Regression model achieved:

**Accuracy: 98.64%**

### Classification Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Fake  | 0.99 | 0.98 | 0.99 |
| Real  | 0.98 | 0.99 | 0.99 |

## 🔄 ML Pipeline

```text
News Input
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
Fake / Real

```

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run app.py
```

### Open in browser

```text
http://localhost:8501
```

## ⚠️ Disclaimer

This application provides an ML-based prediction and does not guarantee
that a news article is factually true or false. Important information
should always be verified using reliable sources.

## 👩‍💻 Author

**Varsha Patel**