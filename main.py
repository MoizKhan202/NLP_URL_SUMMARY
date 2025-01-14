import streamlit as st
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re

# Title of the app
st.title("Fast URL-Based Summarizer 🚀")

# Load a lightweight summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Preprocessing function to clean text
def preprocess_text(text):
    """Clean and preprocess extracted text."""
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces or tabs with a single space
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text)  # Remove non-alphanumeric characters except punctuation
    return text.strip()

# Function to extract text from a URL
def extract_text_from_url(url):
    """Extract and clean text from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()

        # Extract visible text
        raw_text = soup.get_text(separator="\n").strip()

        # Preprocess the extracted text
        cleaned_text = preprocess_text(raw_text)
        return cleaned_text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Sidebar for URL input
st.sidebar.title("Input Section")
url = st.sidebar.text_input("Enter a URL to summarize:")

if url:
    st.info("Extracting and processing text from the URL...")
    context = extract_text_from_url(url)
    if context:
        st.success("Text extracted and cleaned successfully!")
        st.text_area("Extracted Text (Cleaned)", context[:2000], height=300)

        # Generate a summary
        st.info("Generating summary...")
        try:
            summary = summarizer(context, max_length=100, min_length=25, do_sample=False)
            st.header("Summary")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error generating summary: {e}")
