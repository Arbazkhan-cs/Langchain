import numpy as np
import librosa
import tensorflow as tf
import sys

# Define constants (must match training parameters)
SAMPLE_RATE = 16000
MFCC_FEATURES = 13
MAX_LENGTH = 40  # Must match training time padding/truncation

# Load the trained wake word detection model
model = tf.keras.models.load_model("wake_word_model_best.h5")

# Function to extract MFCC features
def extract_mfcc(audio_file):
    """Extract MFCC features from an audio file."""
    try:
        # Load audio
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < MAX_LENGTH:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LENGTH - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LENGTH]
        
        # Reshape for model input
        mfcc = mfcc.reshape(1, MFCC_FEATURES, MAX_LENGTH, 1)
        
        return mfcc
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Function to predict wake word
def predict_wake_word(audio_file):
    """Run wake word detection on an audio file."""
    mfcc = extract_mfcc(audio_file)
    if mfcc is None:
        return

    # Get prediction
    prediction = model.predict(mfcc, verbose=0)[0][0]

    # Set threshold for wake word detection
    if prediction > 0.50:
        print(f"\n🔊 Wake word detected! (Confidence: {prediction * 100:.2f}%)")
    else:
        print(f"Wake word not detected. (Confidence: {prediction * 100:.2f}%)")

# Run prediction on an audio file provided as argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_wakeword.py <audio_file.wav>")
    else:
        audio_path = sys.argv[1]
        predict_wake_word(audio_path)
