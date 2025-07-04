
import streamlit as st
import pickle

# Load models and vectorizer
with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("naive_bayes_model.pkl", "rb") as f:
    naive_bayes_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Title and instructions
st.title("📝 Sentiment Analysis App")
st.markdown("Enter a product review and choose a model to classify it as **Positive**, **Negative**, or **Neutral**.")

# User input
review = st.text_area("✏️ Enter your review below:")

# Model selection
model_choice = st.selectbox("🔍 Choose a model", ["Logistic Regression", "Naive Bayes"])

# Predict button
if st.button("Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Preprocess input
        vector_input = tfidf_vectorizer.transform([review])

        # Make prediction
        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(vector_input)[0]
        else:
            prediction = naive_bayes_model.predict(vector_input)[0]

        # Decode label
        sentiment = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Predicted Sentiment: **{sentiment.upper()}**")
