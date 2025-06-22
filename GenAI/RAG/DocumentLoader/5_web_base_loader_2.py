from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load model
model = ChatGroq(model="llama3-8b-8192", temperature=0.7)

# define prompt template
prompt = PromptTemplate(
    template="Answer to the user question: {question}\n From the following data \n\n{data}",
    input_variables=["question", "data"]
)

# define output parser
parser = StrOutputParser()

# create runnablelambda for url processing
def extract_data(inputs):
    url = inputs["url"]
    loader = WebBaseLoader(web_path=url)
    document = loader.load()
    return {"question": inputs["question"], "data": document[0].page_content}
process_url = RunnableLambda(extract_data)

# create chain
chain = process_url | prompt | model | parser

# run chain
result = chain.invoke({
    "question": "what is the hardware specification of the product",
    "url": "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"
})

print(result)
