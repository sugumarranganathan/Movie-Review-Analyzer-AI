import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load Dataset
df = pd.read_csv("IMDB Dataset.csv", engine="python", on_bad_lines="skip")

# Features and Target
X = df["review"]
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train)

# Model
model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

# Prediction Function
def predict_sentiment(review):

    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction == "positive":
        return f"😊 Positive Review\n\nConfidence: {confidence}%"
    else:
        return f"😞 Negative Review\n\nConfidence: {confidence}%"

# Gradio UI
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter Movie Review Here..."
    ),
    outputs=gr.Textbox(),
    title="🎬 Movie Review Sentiment Analysis",
    description="TF-IDF Vectorizer + Multinomial Naive Bayes"
)

demo.launch()
