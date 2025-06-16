import os
import librosa
import numpy as np
import librosa.display
import soundfile as sf
import random
from tqdm import tqdm

# Input & Output Directories
input_dir = "habibi/"  # Folder with 40 recordings
output_dir = "new_habibi/"  # Folder for 2000 new samples
os.makedirs(output_dir, exist_ok=True)

# Number of new samples per original file
augmentations_per_file = 50  # (40 * 50 = 2000 total)

# Audio settings
TARGET_SR = 16000  # 16kHz Sample Rate
TARGET_LENGTH = TARGET_SR  # 1 second = 16000 samples

def change_pitch(audio, sr, n_steps):
    """Shift the pitch of the audio"""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def change_speed(audio, factor):
    """Change speed while preserving pitch"""
    return librosa.effects.time_stretch(audio, rate=factor)

def add_noise(audio, noise_level=0.005):
    """Add random noise to audio"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def change_volume(audio, gain):
    """Increase or decrease volume"""
    return audio * gain

def time_shift(audio, shift_max=0.2):
    """Randomly shift audio forward or backward"""
    shift = int(random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def fix_length(audio):
    """Ensure audio is exactly 1 second (16,000 samples)"""
    if len(audio) > TARGET_LENGTH:
        return audio[:TARGET_LENGTH]  # Trim if longer
    return np.pad(audio, (0, TARGET_LENGTH - len(audio)), mode='constant')  # Pad if shorter

# Process each original audio file
for file_name in tqdm(os.listdir(input_dir)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(file_path, sr=TARGET_SR)  # Load with 16kHz

        for i in range(augmentations_per_file):
            augmented_audio = audio.copy()

            # Randomly apply augmentations
            if random.choice([True, False]):
                augmented_audio = change_pitch(augmented_audio, sr, random.uniform(-1, 1))
            if random.choice([True, False]):
                augmented_audio = change_speed(augmented_audio, random.uniform(0.8, 1.2))
            if random.choice([True, False]):
                augmented_audio = add_noise(augmented_audio, random.uniform(0.002, 0.01))
            if random.choice([True, False]):
                augmented_audio = change_volume(augmented_audio, random.uniform(0.5, 1.5))
            if random.choice([True, False]):
                augmented_audio = time_shift(augmented_audio)

            # Ensure the audio is exactly 1 second (16,000 samples)
            augmented_audio = fix_length(augmented_audio)

            # Save the augmented file
            output_file = os.path.join(output_dir, f"aug_{file_name[:-4]}_{i}.wav")
            sf.write(output_file, augmented_audio, sr)

print(f"✅ Audio augmentation complete! {augmentations_per_file * len(os.listdir(input_dir))} new samples generated.")
