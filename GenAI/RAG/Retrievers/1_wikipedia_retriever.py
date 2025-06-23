from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    lang="en",
    top_k_results=3
)

query = "Iran vs Israiel"

result = retriever.invoke(query)

print(result)
