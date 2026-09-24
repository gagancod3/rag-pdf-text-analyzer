from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()  # Load environment variables from .env file

pdf_path = Path(__file__).parent / "SQL-Manual.pdf"

# Load the PDF file using PyPDFLoader

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# Split the loaded documents into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=40
)

chunks = text_splitter.split_documents(docs)

# Embedding the chunks using OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Create a vector store using Qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    collection_name="sample_collection",
    url="http://localhost:6333"  # Qdrant server URL
)

print("PDF text has been processed and stored in the vector database.")
