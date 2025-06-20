# import neccessary libraries
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# initilialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# define output_parser
class Info(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    email: EmailStr = Field(description="Email of the person")

parser = PydanticOutputParser(pydantic_object=Info)

# define prompt_template
prompt = PromptTemplate(
    template="Provide me name, age, email of an {country_name} person \n {format_instruction}",
    input_variables=["country_name"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
print(prompt.invoke({"country_name": "American"}))

# define and run chain
chain = prompt | model | parser
result = chain.invoke({"country_name": "American"})

# print result
print(result)
print(result.name)
print(type(result))
