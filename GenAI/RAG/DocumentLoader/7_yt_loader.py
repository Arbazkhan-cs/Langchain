from langchain_community.document_loaders import YoutubeLoader

# Load transcript of a YouTube video
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=QsYGlZkevEg",
    add_video_info=True,
    language=["en", "id"],
    translation="en",
)

docs = loader.load()

print(docs[0].page_content)  
