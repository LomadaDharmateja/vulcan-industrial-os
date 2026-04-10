import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the PDF
PDF_PATH = "data/WEG-WMO-Installation-Operation-and-Maintenance-Manual-of-Electric-Motors.pdf"
DB_DIR = "data/chroma_db"

def index_manual():
    print("📖 Reading the WEG Motor Manual...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2. Split the text into manageable chunks
    # We use 1000 characters with a small overlap so no info is lost in the "cracks"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 3. Create Embeddings (This uses your RTX 3050 Ti's power!)
    print("🧠 Turning text into vectors (Local Embedding)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save to ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"✅ Success! Manual indexed into {DB_DIR}")

if __name__ == "__main__":
    index_manual()