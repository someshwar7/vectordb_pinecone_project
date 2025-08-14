import os
import json
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# 1️⃣ Load environment variables from .env
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX", "book-embeddings")

# -------------------------
# 2️⃣ Embedding model
# -------------------------
# all-mpnet-base-v2 outputs 768 dimensions
model = SentenceTransformer("all-mpnet-base-v2")
dimension = model.get_sentence_embedding_dimension()  # should be 768

# -------------------------
# 3️⃣ Init Pinecone
# -------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if not exists
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)

# -------------------------
# 4️⃣ Text splitter
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# -------------------------
# 5️⃣ Load and process JSONL
# -------------------------
JSONL_PATH = r"C:\Users\somes\Desktop\PROJECT\vector_db_project\data\processed\text_chunks.jsonl"

if not os.path.exists(JSONL_PATH):
    raise FileNotFoundError(f"JSONL file not found at: {JSONL_PATH}")

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue  # skip empty lines

        chunk = json.loads(line)
        text_chunks = splitter.split_text(chunk.get("content", ""))

        for i, part in enumerate(text_chunks):
            vector = model.encode(part).tolist()
            metadata = {
                "page": chunk.get("page_number"),
                "source": chunk.get("source_id")
            }
            index.upsert([
                (f"{chunk['id']}_{i}", vector, metadata)
            ])

print("✅ Book embeddings uploaded successfully!")
