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
            chat_bot = prompt | llm
            response = chat_bot.invoke({"input": user_input, "context": context})
            formatted_response = f"**Your Response:** {response.content}"

            # Append user input and AI response to history
            st.session_state['history'].append((user_input, formatted_response))

            # Convert AI response to speech
            tts = gTTS(text=response.content, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                st.audio(f.name, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Record audio using sounddevice
def record_audio(duration, sample_rate, stop_event):
    recording = []
    
    def callback(indata, frames, time, status):
        if stop_event.is_set():
            return
        recording.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        stop_event.wait(timeout=duration)
    
    if recording:
        st.session_state['recording_data'] = np.concatenate(recording)
    else:
        st.session_state['recording_data'] = None

# Save recorded audio to a WAV file
def save_wav_file(audio_data, sample_rate):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav.write(temp_file.name, sample_rate, audio_data)
        return temp_file.name

# Convert recorded audio to text
def recognize_speech_from_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as e:
        st.error(f"Speech Recognition failed: {str(e)}")
        return None

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
    if 'stop_event' not in st.session_state:
        st.session_state['stop_event'] = threading.Event()
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
            st.session_state['stop_event'].clear()
            st.session_state['recording_thread'] = threading.Thread(
                target=lambda: record_audio(10, 44100, st.session_state['stop_event'])
            )
            st.session_state['recording_thread'].start()
            st.write("Recording started...")
        else:
            st.write("Recording already in progress.")
    
    if st.button("Stop Recording"):
        if st.session_state['recording_thread'] is not None:
            st.session_state['stop_event'].set()
            st.session_state['recording_thread'].join()
            st.session_state['recording_thread'] = None
            st.write("Recording stopped.")
            
            if st.session_state['recording_data'] is not None:
                audio_file = save_wav_file(st.session_state['recording_data'], 44100)
                
                speech_text = recognize_speech_from_audio(audio_file)
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
