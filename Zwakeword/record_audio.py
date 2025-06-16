import pyaudio
import wave
import time

# Audio settings
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono
RATE = 16000  # Sample rate (16kHz)
CHUNK = 1024  # Buffer size
DURATION = 1  # Record time in seconds
NUM_RECORDINGS = 40  # Number of recordings
WAIT_TIME = 1  # Extra wait time after each recording (in seconds)

# Initialize PyAudio
p = pyaudio.PyAudio()

print("Recording will start now...")

for i in range(19, NUM_RECORDINGS + 1):
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print(f"Recording {i}/20...")

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Save the audio file
    filename = f"Habibi/recording_{i}.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"Saved: {filename}")

    # Wait for 1 extra second before the next recording
    time.sleep(WAIT_TIME)

# Terminate PyAudio
p.terminate()

print("Recording completed! 🎤")
