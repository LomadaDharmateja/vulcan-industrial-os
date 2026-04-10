import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# This uses your NVIDIA GPU to turn text into math!
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_knowledge_base(pdf_path):
    """Loads a PDF and saves it into a local Vector Database."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Create the database locally on your disk
    vector_db = Chroma.from_documents(
        documents=pages, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("✅ Knowledge Base built. AI can now 'read' the manual.")
    return vector_db