from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# 1. Load the PDF
loader = PyPDFLoader("Safe-Money-Income-Secrets.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# 2. Chunk it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# 3. Embed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk.page_content for chunk in chunks]
embeddings = embedder.encode(texts).tolist()
print(f"Generated {len(embeddings)} embeddings")

# 4. Store in ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(texts))]
)

print(f"✅ Indexed {collection.count()} chunks. Done.")