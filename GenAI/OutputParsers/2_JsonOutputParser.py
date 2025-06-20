from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# laod environmental variables
load_dotenv()

# initialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# define parser
parser = JsonOutputParser()

# define prompt_template
promptTemplate = PromptTemplate(
    template="Provide me name, age, email of an {country_name} person \n {format_instruction}",
    input_variables=["country_name"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
prompt = promptTemplate.invoke({"country_name": "American"})
print(prompt) # Output text='Provide me name, age, email of an American person \n Return a JSON object.'

# Initialize chain 
chain = promptTemplate | model | parser
result = chain.invoke({"country_name": "American"})
print(result)
print(result["name"])
print(type(result))