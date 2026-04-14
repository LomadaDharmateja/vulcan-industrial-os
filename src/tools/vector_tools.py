import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_vectorstore():
    """Connects to the Pinecone cloud database."""
    
    # 1. Use the API for lightweight queries (0MB RAM!)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Must match local model
    )
    
    # 2. Connect to the Pinecone database you already filled
    vectorstore = PineconeVectorStore(
        index_name="vulcan-manuals",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    return vectorstore