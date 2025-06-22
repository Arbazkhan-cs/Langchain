from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Define prompt template
prompt = PromptTemplate(
    template="Answer the user question: {question} \n\nfrom the context below:\n\n{context}",
    input_variables=["question", "context"]
)

# Define output parser
parser = StrOutputParser()

# Create a RunnableLambda for processing the PDF
def process_file(inputs):
    file = inputs["file"]
    loader = PyPDFLoader(file)
    documents = loader.lazy_load() # load document 1 by 1 and deletes after another from the memory
    context = ""
    for doc in documents:
        context += doc.page_content

    return {"context": context, "question": inputs["question"]}
load_pdf = RunnableLambda(process_file)

# Build chain
chain = load_pdf | prompt | model | parser

# Run the chain
result = chain.invoke({
    "file": "Who are ai engineers.pdf",
    "question": "What are two aspects of generative AI?"
})

print(result)
