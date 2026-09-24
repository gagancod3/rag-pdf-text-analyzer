from fastapi import APIRouter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI


# Initialize the API router
router = APIRouter(
    prefix="/query",
    tags=["query"],
)

# Initialize the OpenAI client
openai_client = OpenAI(
    api_key="your-api-key"  # Replace with your actual OpenAI API key
)  

# Embedding the chunks using OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Create a vector store using Qdrant - connection to vector database
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_model,
    collection_name="sample_collection",
    url="http://localhost:6333"  # Qdrant server URL
)


# Define the query endpoint
@router.post("/")
async def query_text(question: str):

    # Perform a similarity search in the vector database
    vector_results = vector_db.similarity_search(question)
    print(f"Vector search results: {vector_results}")

    context = " ".join(f'[Page {result.metadata.get("page_number", "N/A")}] {result.page_content}' for result in vector_results)

    SYSTEM_PROMPT = """
                    You are a helpful assistant that answers questions based on the provided context.
                    If the context does not contain the answer, respond with 'I don't know.'
                    Also include the page number of the source in your answer if applicable.
                    context: {context}
                    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    
    return {"question": question, "answer": response.choices[0].message.content}
