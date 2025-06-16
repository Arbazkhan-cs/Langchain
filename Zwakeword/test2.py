import pyaudio
import numpy as np
import tensorflow.lite as tflite
import librosa
import time

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="extracted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Audio recording settings
TARGET_SR = 16000  # Sample rate
CHUNK_DURATION = 1  # Seconds
CHUNK_SIZE = TARGET_SR * CHUNK_DURATION  # Total samples in 1 sec
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = TARGET_SR

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

def preprocess_audio_from_array(audio_array, target_sr=16000):
    """Preprocess raw audio data from microphone input."""
    # Convert int16 to float32
    audio = np.array(audio_array, dtype=np.float32) / np.iinfo(np.int16).max  
    
    # Normalize
    audio = audio - np.mean(audio)
    if np.max(audio) > 0:
        audio = audio / np.max(audio)

    # Create spectrogram
    spectrogram = librosa.stft(audio, n_fft=400, hop_length=160)
    spectrogram = np.abs(spectrogram) ** 2  # Convert to power spectrogram
    spectrogram = librosa.power_to_db(spectrogram)  # Convert to log scale
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    
    # Resize to match the TFLite model input shape
    spectrogram = np.resize(spectrogram, (1, 99, 43, 1))  
    return spectrogram.astype(np.float32)  # Convert to float32 for TensorFlow Lite

# Real-time Wake Word Detection Loop
print("Listening for wake word...")
while True:
    audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)  # Read audio
    processed_audio = preprocess_audio_from_array(audio_data)  # Convert to spectrogram
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed_audio)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]  # Get confidence score
    
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:  # Wake word detected
        print(f"Wake word detected! Confidence: {confidence}%")
        print("Hi sir! How can I help you?")  # 🔥 Trigger event
        time.sleep(2)  # Add delay to prevent multiple detections
    else:
        print(f"Wake word NOT detected. Confidence: {confidence}%")

stream.stop_stream()
stream.close()
p.terminate()
