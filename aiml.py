import sys
print("Python executable being used:", sys.executable)

import pdfplumber
import ssl
import spacy

# Bypass SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Increase maximum length for text processing if required
nlp.max_length = 2000000  # Example value, adjust based on available memory

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf('anatomy_vol_1.pdf')

def chunk_text(text, max_length=1000000):
    """Split the text into chunks of max_length characters."""
    chunks = []
    while len(text) > max_length:
        split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        chunks.append(text[:split_point])
        text = text[split_point:].lstrip()
    chunks.append(text)
    return chunks

def preprocess_text(text):
    chunks = chunk_text(text)
    cleaned_sentences = []

    for chunk in chunks:
        doc = nlp(chunk)
        for sent in doc.sents:
            words = [token.text.lower() for token in sent if not token.is_stop and not token.is_punct]
            cleaned_sentences.append(' '.join(words))
    
    return cleaned_sentences

cleaned_text = preprocess_text(pdf_text)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = generate_response("Tell me about the first chapter.")
print(response)

import streamlit as st

st.title("Book Chatbot")
user_input = st.text_input("Ask me anything about the book:")

if user_input:
    response = generate_response(user_input)
    st.write(response)
