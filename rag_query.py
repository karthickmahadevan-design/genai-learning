from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

load_dotenv()

# Load embedder and ChromaDB
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")

# Connect to Claude
claude = anthropic.Anthropic()

def ask(question: str) -> str:
    # Embed the question
    query_embedding = embedder.encode(question).tolist()

    # Retrieve top 3 most relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    chunks = results['documents'][0]

    # Build prompt with retrieved context
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Ask Claude
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Try it
question = "What are the main strategies for generating safe income?"
answer = ask(question)
print(f"\nQ: {question}")
print(f"\nA: {answer}")