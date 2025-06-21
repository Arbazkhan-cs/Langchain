from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="C:\\Users\\Arbaz Khan\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2"
)

text = "Hi, How are you?"

embeddings = embedding.embed_query(text)

print(str(embeddings))