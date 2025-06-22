from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# define document loader
loader = PyPDFLoader("Who are ai engineers.pdf")

# define splitter object
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=90
)

# load document and split it into chunks
documents = loader.load()

documents_chunks = splitter.split_documents(documents)

print(len(documents_chunks))
print("\n", "="*50, "\n")
print(documents_chunks)
print("\n", "="*50, "\n")
print(documents_chunks[0].page_content)

