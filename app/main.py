from fastapi import FastAPI
from routes.query import router as query_router

app = FastAPI(title="RAG PDF Text Analyzer", 
              description="A FastAPI text-analyzer app for PDF files using RAG.", 
              version="1.0.0")

app.include_router(query_router)

