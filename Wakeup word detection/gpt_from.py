import speech_recognition as sr
from langchain_groq import ChatGroq
import pyttsx3
from dotenv import load_dotenv
import threading
import queue
import time

load_dotenv()

class VoiceAssistant:
    def __init__(self, wake_word="Habibi"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        # Initialize text-to-speech engine once
        self.engine = pyttsx3.init()
        
        # Pre-initialize the Groq client
        self.chat = ChatGroq(model="llama3-8b-8192")
        
        # Optimize speech recognition settings
        self.recognizer.energy_threshold = 300  # Lower energy threshold for faster detection
        self.recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment
        self.recognizer.pause_threshold = 0.5  # Shorter pause threshold
        
    def speak(self, text):
        """Optimized speech function"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def background_listener(self):
        """Continuously listen in background"""
        mic = sr.Microphone()
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while self.is_running:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Background listener error: {e}")
                    time.sleep(0.1)
    
    def process_audio(self):
        """Process audio from queue"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    if self.wake_word in text:
                        print("Wake word detected!")
                        self.speak("Yes, how can I help you?")
                        self.handle_question()
                        
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Processing error: {e}")
            time.sleep(0.1)
    
    def handle_question(self):
        """Handle user question"""
        with sr.Microphone() as source:
            try:
                print("Listening for your question...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                question = self.recognizer.recognize_google(audio)
                print(f"You asked: {question}")
                
                # Process response in separate thread
                thread = threading.Thread(target=self.process_response, args=(question,))
                thread.start()
                
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't catch that.")
            except sr.RequestError:
                self.speak("There was an error with the speech service.")
    
    def process_response(self, question):
        """Process LLM response in separate thread"""
        try:
            response = self.chat.invoke(question).content
            print(f"Assistant: {response}")
            self.speak(response)
        except Exception as e:
            print(f"LLM error: {e}")
            self.speak("Sorry, I encountered an error processing your question.")
    
    def start(self):
        """Start the voice assistant"""
        print(f"Starting voice assistant... Wake word is '{self.wake_word}'")
        
        # Start background listening thread
        listener_thread = threading.Thread(target=self.background_listener)
        listener_thread.daemon = True
        listener_thread.start()
        
        # Start processing thread
        processor_thread = threading.Thread(target=self.process_audio)
        processor_thread.daemon = True
        processor_thread.start()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant"""
        self.is_running = False
        print("\nStopping voice assistant...")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.start()