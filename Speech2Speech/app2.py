from flask import Flask, jsonify, send_file, render_template
import sounddevice as sd
import numpy as np
import threading
import tempfile
import speech_recognition as sr
from gtts import gTTS
import wave
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

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

    Previous Context: {previous_context}

    User Input: {input}
    """
    return ChatPromptTemplate.from_template(template)

# Global variables for recording and context
recording_data = []
stop_event = threading.Event()
recording_thread = None
previous_contexts = []

def record_audio(sample_rate=44100):
    global recording_data
    recording_data = []
    stop_event.clear()

    def callback(indata, frames, time, status):
        if stop_event.is_set():
            raise sd.CallbackStop
        recording_data.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        stop_event.wait()

    return np.concatenate(recording_data) if recording_data else None

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

def generate_audio_response(text):
    """Generate an audio file from the text."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        temp_file.close()  # Close the file so it can be read later
        return temp_file.name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_thread
    if recording_thread and recording_thread.is_alive():
        return jsonify({"status": "Recording already in progress."}), 400

    stop_event.clear()
    recording_thread = threading.Thread(target=lambda: record_audio())
    recording_thread.start()
    return jsonify({"status": "Recording started."})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    if not recording_thread or not recording_thread.is_alive():
        return jsonify({"status": "No recording in progress."}), 400

    stop_event.set()
    recording_thread.join()

    if recording_data:
        wav_file_path = save_wav_file(np.concatenate(recording_data), 44100)
        text = convert_audio_to_text(wav_file_path)
        
        model = load_model()
        
        prompt = create_prompt_template()
        
        if len(previous_contexts) > 1:
            previous_context = f"{previous_contexts[-2]} {previous_contexts[-1]}"
        else:
            previous_context = previous_contexts[-1] if previous_contexts else ""

        formatted_prompt = prompt.format(previous_context=previous_context, input=text)
        
        print("Formatted Prompt:", formatted_prompt)
        
        try:
            chatbot_response = model.invoke(formatted_prompt)
            # print("Chatbot Response:", chatbot_response)
            
            # Update context history
            previous_contexts.append(text + " " + chatbot_response.content)
            if len(previous_contexts) > 2:
                previous_contexts.pop(0)
            
            # Generate audio response
            audio_response_path = generate_audio_response(chatbot_response.content)
            return send_file(audio_response_path, as_attachment=True)
        
        except Exception as e:
            print("Error generating chatbot response:", e)
            return jsonify({"status": "Error generating response."}), 500

    else:
        return jsonify({"status": "No recording data available."}), 500

if __name__ == '__main__':
    app.run(debug=False)