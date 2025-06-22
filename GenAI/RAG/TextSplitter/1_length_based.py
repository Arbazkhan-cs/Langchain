from langchain.text_splitter import CharacterTextSplitter

text = """
Cats are fascinating creatures known for their independence, agility, and quiet charm. With their sleek fur, sharp eyes, and graceful movements, they often seem like little hunters living among us. Whether they're leaping onto high shelves or silently stalking a toy mouse, their playful instincts are always on display. Cats are curious by nature and love to explore their surroundings, often finding the coziest or most unexpected places to nap.

Despite their independent streak, cats form strong bonds with their owners. They may not always show affection in obvious ways, but a soft purr, a gentle head bump, or simply choosing to sit beside you speaks volumes. Their presence can be calming and comforting, especially after a long day. A cat's companionship is a unique blend of mystery and warmth, making them beloved pets around the world.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=""
)

doc = splitter.split_text(text)

print(len(doc))
print(doc)
