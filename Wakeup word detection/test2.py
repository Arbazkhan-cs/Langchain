import os
import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import time
import threading
from queue import Queue

# Load your trained model
model = tf.keras.models.load_model("wake_word_model_best.h5")

# Audio parameters (should match your training parameters)
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
RECORD_SECONDS = 2  # Sliding window size
MFCC_FEATURES = 13
MAX_LENGTH = 40  # Should match your training data

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Function to extract MFCC features
def extract_mfcc(audio_data, sample_rate=SAMPLE_RATE, max_length=MAX_LENGTH):
    """Extract MFCC features from audio data."""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=MFCC_FEATURES)
    
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    
    return mfcc

# Queue for audio processing
audio_queue = Queue()
is_running = True
wake_word_detected = False  # Flag to prevent repeated detections

# Function to continuously record audio
def record_audio():
    stream = audio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
    
    print("Listening for wake word...")
    
    sliding_window = np.zeros(SAMPLE_RATE * RECORD_SECONDS)
    
    try:
        while is_running:
            # Read audio chunk
            data = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.float32)
            
            # Update sliding window
            sliding_window = np.append(sliding_window[len(data):], data)
            
            # Add to queue for processing
            audio_queue.put(sliding_window.copy())
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()

# Function to process audio and detect wake word
def process_audio():
    global wake_word_detected
    cooldown_time = 2  # Cooldown period in seconds
    last_detection_time = 0

    while is_running:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            
            # Extract features
            mfcc = extract_mfcc(audio_data)
            
            # Reshape for model input (batch, height, width, channels)
            mfcc = mfcc.reshape(1, MFCC_FEATURES, MAX_LENGTH, 1)
            
            # Predict
            prediction = model.predict(mfcc, verbose=0)[0][0]
            current_time = time.time()

            if prediction > 0.85 and not wake_word_detected:
                wake_word_detected = True  # Set flag
                print("\n🔊 Wake word detected! (Confidence: {:.2f}%)".format(prediction * 100))
                print("Hello Sir!")

                # Update last detection time
                last_detection_time = current_time
                
                # Cooldown to reset wake word detection
                threading.Timer(cooldown_time, reset_wake_word).start()
            else:
                print("Wake word not detected. (Confidence: {:.2f}%)".format((1 - prediction) * 100))

        # Small sleep to prevent excessive CPU usage
        time.sleep(0.5)  # Increased time to prevent spam

# Function to reset wake word detection flag
def reset_wake_word():
    global wake_word_detected
    wake_word_detected = False

# Create and start threads
record_thread = threading.Thread(target=record_audio)
process_thread = threading.Thread(target=process_audio)

try:
    record_thread.start()
    process_thread.start()
    
    # Keep the main thread alive
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopping wake word detection...")
    is_running = False
    
    # Wait for threads to finish
    record_thread.join()
    process_thread.join()
    
    # Clean up PyAudio
    audio.terminate()
    print("Wake word detection stopped.")