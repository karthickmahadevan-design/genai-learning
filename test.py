from dotenv import load_dotenv
import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# Test 1 — environment variables
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
print(f"✅ Anthropic key loaded: {anthropic_key[:15]}...")
print(f"✅ OpenAI key loaded: {openai_key[:15]}...")

# Test 2 — ChromaDB
client = chromadb.Client()
col = client.create_collection("test")
col.add(documents=["hello world"], ids=["1"])
print("✅ ChromaDB working")

# Test 3 — Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
vec = model.encode("test sentence")
print(f"✅ Embeddings working — vector dim: {len(vec)}")

# Test 4 — Claude API
claude = anthropic.Anthropic()
resp = claude.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say: stack verified"}]
)

print(f"✅ Claude API: {resp.content[0].text}")


# Test 5 — OpenAI API
import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",  # cheapest, good enough for learning
    messages=[{"role": "user", "content": "Say: OpenAI stack verified"}],
    max_tokens=20
)

print(type(response))
print(response)

print(f"✅ OpenAI API: {response.choices[0].message.content}")




print("\n── All systems go. Ready to build. ──")