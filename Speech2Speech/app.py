import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import tempfile
import threading
from gtts import gTTS
import wave

# Function to load language model
def load_model():
    """Load the language model."""
    load_dotenv()
    return ChatGroq(model="llama3-8b-8192", temperature=0.7)

# Create the prompt template
def create_prompt_template():
    """Create the prompt template for the interviewer."""
    template = """
    You are a helpful AI English Interviewer and Feedback Assistant. Your role is to simulate an interview with the user, ask thoughtful questions, and provide constructive feedback on their responses to help them improve their English skills.

    Your communication should be friendly, encouraging, and focused on:
    - Asking relevant questions.
    - Offering detailed feedback on fluency.
    - Providing tips on how the user can enhance their responses.

    Context:
    {context}

    User Input: {input}
    """
    return ChatPromptTemplate.from_template(template)

# Get last two interactions from history for context
def get_last_two_history():
    history = st.session_state.get('history', [])
    if len(history) >= 2:
        last_two = history[-2:]
    else:
        last_two = history
    context = "\n".join([f"**User Input:** {user_input}\n**AI Response:** {ai_response}"
                         for user_input, ai_response in last_two])
    return context

# Display chat history
def display_conversation_history():
    if st.session_state['history']:
        st.subheader("Conversation History")
        for user_input, ai_response in reversed(st.session_state['history']):
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI:** {ai_response}")

# Handle text or audio input, invoke model, and store responses in history
def handle_user_input(prompt, llm, user_input=None):
    if user_input:
        context = get_last_two_history()
        try:
            response = llm({"input": user_input, "context": context})
            formatted_response = f"**Your Response:** {response['content']}"

            # Append user input and AI response to history
            st.session_state['history'].append((user_input, formatted_response))

            # Convert AI response to speech
            tts = gTTS(text=response['content'], lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                st.audio(f.name, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")

stop_event = threading.Event()

# Record audio using sounddevice
def record_audio(sample_rate=44100):
    st.session_state["recording_data"] = []
    stop_event.clear()

    def callback(indata, frames, time, status):
        if stop_event.is_set():
            raise sd.CallbackStop
        st.session_state["recording_data"].append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.wait()

    return np.concatenate(st.session_state["recording_data"]) if st.session_state["recording_data"] else None


def save_wav_file(audio_data, sample_rate):
    """Save audio data as a WAV file."""
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # sample width in bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        temp_file.close()  # Close the file so it can be read later
        return temp_file.name

def convert_audio_to_text(audio_file):
    """Convert audio file to text."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results; check your network connection"

# Clear conversation history
def clear_history():
    if st.button("Clear History"):
        st.session_state['history'] = []
        st.experimental_rerun()

# Main function to run the app
def main():
    llm = load_model()
    prompt = create_prompt_template()

    # Streamlit app setup
    st.title("AI English Interviewer and Feedback Assistant")
    st.write("Practice your English with interactive feedback from an AI interviewer.")

    # Initialize session state for chat history and recording
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'recording_thread' not in st.session_state:
        st.session_state['recording_thread'] = None
    if 'recording_data' not in st.session_state:
        st.session_state['recording_data'] = None

    # Display conversation history
    display_conversation_history()

    # Handle text input
    user_input = st.text_input("Your response:")
    if user_input:
        handle_user_input(prompt, llm, user_input=user_input)

    # Handle audio input
    st.header("Record your response")
    if st.button("Start Recording"):
        if st.session_state['recording_thread'] is None:
            stop_event.clear()
            with st.spinner('Recording...'):
                st.session_state['recording_thread'] = threading.Thread(
                    target=lambda: record_audio()
                )
                st.session_state['recording_thread'].start()
            st.write("Recording started...")
        else:
            st.write("Recording already in progress.")
    
    if st.button("Stop Recording"):
        if st.session_state['recording_thread'] is not None:
            stop_event.set()
            with st.spinner('Stopping recording...'):
                st.session_state['recording_thread'].join()
            st.session_state['recording_thread'] = None
            st.write("Recording stopped.")
            
            if st.session_state['recording_data']:
                audio_file = save_wav_file(st.session_state['recording_data'], 44100)
                speech_text = convert_audio_to_text(audio_file)
                if speech_text:
                    st.write(f"Recognized Text: {speech_text}")
                    handle_user_input(prompt, llm, user_input=speech_text)
            else:
                st.write("No recording data available.")
        else:
            st.write("No recording in progress.")

    # Clear conversation history
    clear_history()

if __name__ == "__main__":
    main()
