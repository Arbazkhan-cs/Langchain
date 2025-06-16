import speech_recognition as sr
import time
from langchain_groq import ChatGroq
from queue import Queue
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

class WakeWordDetector:
    def __init__(self, wake_word="hey assistant"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.audio_queue = Queue()
        
        # Initialize Groq client
        self.llm = ChatGroq(model="llama3-8b-8192")
        
    def listen_in_background(self):
        """Continuously listen for audio in the background"""
        def callback(recognizer, audio):
            self.audio_queue.put(audio)
        
        self.stop_listening = self.recognizer.listen_in_background(
            sr.Microphone(), callback)
        self.is_listening = True
    
    def process_audio(self):
        """Process audio from the queue and detect wake word"""
        while self.is_listening:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    # Convert audio to text
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    # Check for wake word
                    if self.wake_word in text:
                        print("Wake word detected!")
                        print("Waiting for 2 seconds before listening to your question...")
                        
                        # Stop background listening while handling the question
                        self.stop_listening(wait_for_stop=True)
                        
                        # Wait for 2 seconds
                        time.sleep(2)
                        
                        # Handle the question
                        self.handle_question()
                        
                        # Small pause before resuming wake word detection
                        time.sleep(0.5)
                        
                        # Resume background listening for wake word
                        self.listen_in_background()
                        print(f"\nGoing back to sleep mode. Listening for wake word: '{self.wake_word}'")
                        
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
    
    def handle_question(self):
        """Listen for and handle a single question"""
        with sr.Microphone() as source:
            print("Now listening for your question...")
            try:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for the question with a timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
                question = self.recognizer.recognize_google(audio)
                print(f"\nYour question: {question}")
                
                # Get response from Groq
                print("Processing your question...")
                response = self.get_groq_response(question)
                print(f"\nAssistant: {response}")
                
            except sr.WaitTimeoutError:
                print("No question detected within timeout period. Going back to sleep mode.")
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the question. Going back to sleep mode.")
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
    
    def get_groq_response(self, question):
        """Get response from Groq API"""
        try:
            completion = self.llm.invoke(question).content
            return completion
        except Exception as e:
            return f"Error getting response from Groq: {str(e)}"
    
    def start(self):
        """Start the wake word detection system"""
        self.listen_in_background()
        print(f"System initialized and in sleep mode. Listening for wake word: '{self.wake_word}'")
        
        # Start processing audio in a separate thread
        process_thread = Thread(target=self.process_audio)
        process_thread.start()
        
    def stop(self):
        """Stop the wake word detection system"""
        if self.is_listening:
            self.stop_listening(wait_for_stop=False)
            self.is_listening = False

def main():    
    # Initialize and start the wake word detector
    detector = WakeWordDetector(wake_word="hey assistant")
    
    try:
        detector.start()
        # Keep the main thread running
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping the assistant...")
        detector.stop()

if __name__ == "__main__":
    main()