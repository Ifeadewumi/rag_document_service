from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import documents, query

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RAG Document Search API",
    description="AI-powered document search with RAG using vector embeddings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/")
async def root():
    return {
        "message": "RAG Document Search API",
        "endpoints": {
            "upload": "POST /documents/upload",
            "list": "GET /documents",
            "detail": "GET /documents/{id}",
            "query": "POST /query"
        },
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-document-search"
    }