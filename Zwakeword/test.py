import pyaudio
import numpy as np
import librosa
import tensorflow as tf
import time

# Load your trained model
model = tf.keras.models.load_model("fine_tuned_model.h5")  # Change path if needed

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
    spectrogram = tf.signal.stft(audio, frame_length=400, frame_step=160)
    spectrogram = tf.abs(spectrogram) ** 2  # Convert to power spectrogram
    spectrogram = tf.image.resize(spectrogram[..., tf.newaxis], [40, 101])  # Resize

    spectrogram = np.log10(spectrogram + 1e-6)  # Log transform
    return np.reshape(spectrogram, (1, 40, 101, 1))  # Model input shape

# Real-time Wake Word Detection Loop
print("Listening for wake word...")
while True:
    audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)  # Read audio
    processed_audio = preprocess_audio_from_array(audio_data)  # Convert to spectrogram
    
    prediction = model.predict(processed_audio)[0][0]  # Get confidence score
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:  # Wake word detected
        print(f"Wake word detected! Confidence: {confidence}%")
        print("Hi sir! How can i help you")  # 🔥 Trigger event
        time.sleep(2)  # Add delay to prevent multiple detections

stream.stop_stream()
stream.close()
p.terminate()
