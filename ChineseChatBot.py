
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st



pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")


# Load model directly

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)

st.title("Amn's Mandarin Chatbot")
user_input = st.text_area("Enter your text here: ")



if st.button("Go"):
    if user_input:
        translated_texts = translation_pipeline(user_input)
        translated_text = translated_texts[0]['translation_text']
        st.write("Response:")
        st.write(translated_text)