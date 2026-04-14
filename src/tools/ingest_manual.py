import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load the API keys from your local .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    raise ValueError("Missing API Keys! Check your .env file.")

def ingest_document():
    print("🚀 Starting Offline Ingestion Pipeline...")

    # 2. Load the WEG Manual
    # (Make sure this filename matches exactly what is in your data folder!)
    file_path = "data/WEG-WMO-Installation-Operation-and-Maintenance-Manual-of-Electric-Motors.pdf"
    print(f"📄 Loading manual from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 3. Chunk the text so the AI can read it in pieces
    print("✂️ Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks.")

    # 4. Connect to HuggingFace Cloud API (0MB local RAM used!)
    print("🧠 Connecting to HuggingFace Inference API...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Upload the converted numbers directly to Pinecone
    print("☁️ Uploading vectors to Pinecone database... (This might take a minute)")
    index_name = "vulcan-manuals"

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("🎉 Ingestion Complete! The AI Brain is now safely in the cloud.")

if __name__ == "__main__":
    # Note: We use a relative path trick so you can run it from the root folder
    # If the file path fails, adjust the '../data' to just 'data' depending on where you run it.
    ingest_document()