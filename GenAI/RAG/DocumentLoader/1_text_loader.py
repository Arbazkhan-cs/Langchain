from langchain_community.document_loaders import TextLoader 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Define promptTemplate
prompt = PromptTemplate(
    template="Provide the summary of the following text \n {text}",
    input_variables=["text"]
)

# Define output parser
parser = StrOutputParser()

# Define text document loader
loader = TextLoader('funny_cat.txt', encoding='utf-8')

document = loader.load()

print(document) # it is a list of documents
print("="*50, "\n")
print(document[0].page_content)
print("="*50, "\n")
print(document[0].metadata)
print("="*50, "\n")

# create a RunnableLambda for creating the chain
def process_file(inputs):
    file = inputs["file"]
    loader = TextLoader(file, encoding='utf-8')
    document = loader.load()
    return document[0].page_content
load_text = RunnableLambda(process_file)

# create a chain
# chain = prompt | model | parser
final_chain = load_text | prompt | model | parser

# run the chain
# result = chain.invoke({"text": document[0].page_content})
result = final_chain.invoke({"file": "funny_cat.txt"})
print("Summary of the text = " + result)
