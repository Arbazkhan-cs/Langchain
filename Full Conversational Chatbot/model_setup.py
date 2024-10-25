# model_setup.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

def initialize_llm():
    """
    Initializes the language model with API keys and temperature settings.
    """
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("Please set GROQ_API_KEY in your environment variables")
    
    return ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
