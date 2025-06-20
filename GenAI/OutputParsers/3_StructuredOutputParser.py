"""StructuredOutputParser provides good control over the json output, but it does not validate the json output"""

# import neccessary libraries
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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
parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="name", description="name of the person"),
    ResponseSchema(name="age", description="age of the person"),
    ResponseSchema(name="email", description="email of the person")
])

# define prompt_template
prompt = PromptTemplate(
    template="Provide me name, age, email of an {country_name} person \n {format_instruction}",
    input_variables=["country_name"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
print(prompt.invoke({"country_name": "American"}))

# define chain
chain = prompt | model | parser
result = chain.invoke({"country_name": "American"})
print(result)
print(result["email"])
print(type(result))
