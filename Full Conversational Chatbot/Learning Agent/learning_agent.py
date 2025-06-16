import pandas as pd
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import logging
import json
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration settings for the QA agent."""
    
    DEFAULT_CONFIG = {
        "model_name": "llama3-8b-8192",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.5,
        "similarity_threshold": 0.7,
        "max_history": 10
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return {**self.DEFAULT_CONFIG, **config}
        except FileNotFoundError:
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG
    
    def _save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def update_config(self, updates: Dict):
        """Update configuration with new values."""
        self.config.update(updates)
        self._save_config(self.config)

class QADatabase:
    """Manages the storage and retrieval of QA pairs."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_csv_exists()
        
    def _ensure_csv_exists(self):
        """Create CSV file if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=['question', 'answer', 'timestamp']).to_csv(
                self.csv_path, index=False
            )
    
    def add_qa_pair(self, question: str, answer: str):
        """Add a new QA pair to the database."""
        df = pd.read_csv(self.csv_path)
        new_row = pd.DataFrame({
            'question': [question],
            'answer': [answer],
            'timestamp': [datetime.now().isoformat()]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
    
    def get_all_qa_pairs(self) -> pd.DataFrame:
        """Retrieve all QA pairs."""
        return pd.read_csv(self.csv_path)
    
    def search_questions(self, query: str) -> pd.DataFrame:
        """Search for questions containing the query."""
        df = pd.read_csv(self.csv_path)
        return df[df['question'].str.contains(query, case=False, na=False)]

class LearningQAAgent:
    """An AI agent that can learn from QA interactions."""
    
    def __init__(self, csv_path: str, config_path: str = "config.json"):
        """
        Initialize the QA agent with enhanced configuration and error handling.
        
        Args:
            csv_path: Path to the CSV file with QA pairs
            config_path: Path to the configuration file
        """
        load_dotenv()
        
        self.config_manager = ConfigManager(config_path)
        self.database = QADatabase(csv_path)
        
        try:
            self._initialize_components()
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _initialize_components(self):
        """Initialize LangChain components with error handling."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config_manager.config["embedding_model"]
            )
            self.llm = ChatGroq(
                temperature=self.config_manager.config["temperature"],
                model=self.config_manager.config["model_name"]
            )
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.config_manager.config["max_history"]
            )
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def load_knowledge_base(self):
        """Load and initialize the knowledge base with error handling."""
        try:
            df = self.database.get_all_qa_pairs()
            texts = [f"Q: {q}\nA: {a}" for q, a in zip(df['question'], df['answer'])]
            
            if texts:
                self.vectorstore = FAISS.from_texts(texts, self.embeddings)
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": 3, "score_threshold": self.config_manager.config["similarity_threshold"]}
                    ),
                    memory=self.memory,
                    return_source_documents=True
                )
            else:
                logger.warning("No QA pairs found in database")
        except Exception as e:
            logger.error(f"Knowledge base loading failed: {str(e)}")
            raise
    
    def add_to_knowledge_base(self, question: str, answer: str):
        """Add a new QA pair with validation and error handling."""
        try:
            if not question.strip() or not answer.strip():
                raise ValueError("Question and answer cannot be empty")
            
            self.database.add_qa_pair(question, answer)
            text = f"Q: {question}\nA: {answer}"
            self.vectorstore.add_texts([text])
            logger.info(f"Added new QA pair - Q: {question}")
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {str(e)}")
            raise
    
    def get_answer(self, question: str) -> Tuple[Optional[str], bool]:
        """Get answer with improved error handling and logging."""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            result = self.qa_chain({"question": question})
            source_docs = result.get("source_documents", [])
            
            if not source_docs:
                logger.info(f"No answer found for question: {question}")
                return None, False
            
            logger.info(f"Answer found for question: {question}")
            return result["answer"], True
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return None, False
    
    def interact(self):
        """Enhanced interactive loop with additional commands and error handling."""
        print("\nQA Agent: Hello! Ask questions or use these commands:")
        print("- 'quit': Exit the program")
        print("- 'search <term>': Search previous questions")
        print("- 'config': Show current configuration")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("QA Agent: Goodbye!")
                    break
                
                elif user_input.lower().startswith('search '):
                    search_term = user_input[7:].strip()
                    results = self.database.search_questions(search_term)
                    if not results.empty:
                        print("\nFound these related questions:")
                        for _, row in results.iterrows():
                            print(f"Q: {row['question']}")
                            print(f"A: {row['answer']}\n")
                    else:
                        print("No matching questions found.")
                
                elif user_input.lower() == 'config':
                    print("\nCurrent configuration:")
                    for key, value in self.config_manager.config.items():
                        print(f"{key}: {value}")
                
                else:
                    answer, found = self.get_answer(user_input)
                    
                    if not found:
                        print("\nQA Agent: I don't know the answer to that question.")
                        print("Would you like to teach me? (yes/no)")
                        
                        if input().lower() == 'yes':
                            print("Please provide the answer:")
                            new_answer = input().strip()
                            self.add_to_knowledge_base(user_input, new_answer)
                            print("Thank you! I've learned something new.")
                    else:
                        print(f"\nQA Agent: {answer}")
                        
            except Exception as e:
                logger.error(f"Error in interaction: {str(e)}")
                print("An error occurred. Please try again.")

def main():
    """Main function with error handling."""
    try:
        agent = LearningQAAgent(
            csv_path="Test.csv",
            config_path="config.json"
        )
        agent.interact()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print("Failed to start the QA Agent. Check the logs for details.")

if __name__ == "__main__":
    main()