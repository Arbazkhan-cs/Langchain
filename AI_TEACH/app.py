import os
import sqlite3
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import datetime


# Initialize database
def init_db():
    conn = sqlite3.connect('language_learning.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mistakes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  target_language TEXT,
                  known_language TEXT,
                  user_input TEXT,
                  correction TEXT,
                  category TEXT)''')
    conn.commit()
    conn.close()

class LanguageTutor:
    def __init__(self, target_lang, known_lang, proficiency):
        # Initialize LLM 
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
       
        # Initialize memory 
        self.message_history = []
       
        self.scenes = {
            '1': 'restaurant',
            '2': 'airport',
            '3': 'hotel',
            '4': 'shop'
        }
        
        # Store language learning context
        self.target_lang = target_lang
        self.known_lang = known_lang
        self.proficiency = proficiency
        self.current_scene = None
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a {target_lang} language tutor.
            The student knows {known_lang} and is at {proficiency} level.
            Current scenario: {current_scene}.
            Conduct conversation in {target_lang}.
            Correct mistakes in this format:
            Correction: [explanation]
            Response: [actual response]"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
       
        # Create runnable
        self.runnable = self.prompt | self.llm

    def log_mistake(self, user_input, correction, category):
        """Log language learning mistakes to SQLite database"""
        try:
            # Use the existing database connection you've already set up
            conn = sqlite3.connect('language_learning.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mistakes 
                (timestamp, target_language, known_language, user_input, correction, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.datetime.now(), self.target_lang, self.known_lang, user_input, correction, category))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging mistake: {e}")

    def select_scene(self):
        """Select conversation scenario and start the conversation"""
        print("\nSelect a scenario:")
        for key, value in self.scenes.items():
            print(f"{key}. {value.capitalize()}")
       
        choice = input("Enter choice (1-4): ")
        self.current_scene = self.scenes.get(choice, 'restaurant')
       
        # Clear previous message history
        self.message_history = []
       
        # Prepare context for initial greeting
        context = {
            "target_lang": self.target_lang,
            "known_lang": self.known_lang,
            "proficiency": self.proficiency,
            "current_scene": self.current_scene,
            "history": self.message_history,
            "input": "Greet the student and start the conversation"
        }
       
        # Invoke the runnable
        response = self.runnable.invoke(context)
        return response.content

    def chat_loop(self):
        """Main conversation loop"""
        print("\nStart chatting (type 'exit' to end):")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break
           
            # Add user message to history
            self.message_history.append(HumanMessage(content=user_input))
           
            # Prepare context for response
            context = {
                "target_lang": self.target_lang,
                "known_lang": self.known_lang,
                "proficiency": self.proficiency,
                "current_scene": self.current_scene,
                "history": self.message_history,
                "input": user_input
            }
           
            # Get bot response
            response = self.runnable.invoke(context)
            bot_response = response.content
           
            if "Correction:" in bot_response:
                parts = bot_response.split("Response:")
                correction_part = parts[0].replace("Correction:", "").strip()
                response_part = parts[1].strip() if len(parts) > 1 else ""
               
                # Log the mistake
                self.log_mistake(user_input, correction_part, "Grammar")
                
                print(f"\nBot: {response_part}")
               
                # Add assistant message to history
                self.message_history.append(AIMessage(content=response_part))
            else:
                print(f"\nBot: {bot_response}")
                # Add assistant message to history
                self.message_history.append(AIMessage(content=bot_response))

    def generate_report(self):
        """Generate a report of language learning mistakes"""
        try:
            # Use the existing database connection
            conn = sqlite3.connect('language_learning.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT category, COUNT(*) as mistake_count 
                FROM mistakes 
                WHERE target_language = ? AND known_language = ?
                GROUP BY category
            ''', (self.target_lang, self.known_lang))
            
            mistake_summary = cursor.fetchall()
            
            print("\n--- Learning Review ---")
            print(f"Target Language: {self.target_lang}")
            print(f"Proficiency Level: {self.proficiency}")
            
            for category, count in mistake_summary:
                print(f"{category} Mistakes: {count}")
            
            if not mistake_summary:
                print("Great job! No significant mistakes recorded.")
            
            conn.close()
        except Exception as e:
            print(f"Error generating report: {e}")

def main():
    init_db()
    # Collect user preferences
    target_lang = input("What language do you want to learn? ")
    known_lang = input("What is your native language? ")
    proficiency = input("What's your current proficiency level (beginner/intermediate/advanced)? ")

    # Create and run the language tutor
    tutor = LanguageTutor(target_lang, known_lang, proficiency)
    print(tutor.select_scene())
    tutor.chat_loop()
    tutor.generate_report()

if __name__ == "__main__":
    main()