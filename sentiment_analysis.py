#Jay Lawrence C. Cerniaz
#Wian Lei A. Santos
#BSIT - 2A
#Date: 10/14/2025

import streamlit as st
import torch
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠", layout="centered")

# Load the RoBERTa model with a true Neutral class
device = 0 if torch.cuda.is_available() else -1 # Check if GPU is available; otherwise, use CPU, because with GPU it would be faster 

# Load the sentiment analysis model with device optimization
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=device
)

# Title and Subheader
st.markdown(
    """
    <h1 style="text-align: center; color: #6C63FF;">
        AI Sentiment Analyzer 🤖💬
    </h1>
    <p style="text-align: center; font-size: 18px;">
        Enter any text and let AI detect its sentiment! 🚀
    </p>
    """,
    unsafe_allow_html=True
)

# User Input
st.write("### ✨ Type your text below:")
user_input = st.text_area("🔍 Enter your text here:")

# Sentiment Analysis Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        label_map = {
            "LABEL_0": ("NEGATIVE", "😠", "red"),
            "LABEL_1": ("NEUTRAL", "😐", "gray"),
            "LABEL_2": ("POSITIVE", "😊", "green"),
        }

        label, emoji, color = label_map[result["label"]]
        confidence = result["score"]

        st.markdown(
            f"""
            <div style="border-radius: 10px; padding: 20px; text-align: center; background-color: {color}; color: white;">
                <h2>{emoji} Sentiment: {label} </h2>
                <p>Confidence Score: {confidence:.2f} 🔥</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter some text before analyzing!")

# Footer
st.markdown(
    "<hr><p style='text-align: center;'>Made with ❤️ by Jay Lawrence Cerniaz and Wian Lei Santos</p>",
    unsafe_allow_html=True
)
