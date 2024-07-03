from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# load all the environment variables
load_dotenv()

# build/load llm and prompt template
llm = ChatGroq(model="llama3-8b-8192", temperature=0.8, max_tokens=512)
prompt1 = ChatPromptTemplate.from_template("Write an essay on {topic} in 100 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem on {topic} for a 5 year old child")

# build FastAPI app and routes
app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="Langchain server for GROQ models"
)

add_routes(
    app,
    prompt1 | llm,
    path = "/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path = "/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)