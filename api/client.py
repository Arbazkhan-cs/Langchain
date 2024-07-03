import requests
import streamlit as st

def get_essay_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={"input": {"topic": input_text}})
    
    return response.json()['output']['content']

def get_poem_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={"input": {"topic": input_text}})
    
    return response.json()['output']['content']

st.title("Langchain demo with GROQ API")
essay_input = st.text_input("Enter the topic for essay")
if essay_input:
    st.write(get_essay_response(essay_input))

poem_input = st.text_input("Enter the topic for poem")
if poem_input:
    st.write(get_poem_response(poem_input))

