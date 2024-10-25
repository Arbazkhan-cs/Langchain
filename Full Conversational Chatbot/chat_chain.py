# chat_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

def create_retrieval_chain(llm, retriever):
    """
    Creates a conversational retrieval chain with a custom prompt template.
    Args:
        llm: The language model.
        retriever: The retriever object for handling document retrieval.
    Returns:
        chain: Conversational retrieval chain.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a conversational AI chatbot. Engage naturally. 
        If the question relates to content in the provided PDF, respond based on PDF information and mention the source.
        Otherwise, respond using general knowledge.
        """),
        ("human", "{question}\n\nContext:\n{context}")
    ])
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": template, "document_variable_name": "context"},
        verbose=False
    )
