from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import time
from app.database import get_db
from app.config import get_settings
from app.models import DocumentChunk
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["Query"])
settings = get_settings()


@router.post("", response_model=QueryResponse)
async def query_documents(
    query: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Answer a question using RAG.
    
    Process:
    1. Embed the user's question
    2. Search vector DB for relevant chunks
    3. Build RAG prompt with retrieved context
    4. Generate answer using LLM
    5. Return answer with source chunks and scores
    """
    start_time = time.time()
    
    try:
        # Step 1: Generate query embedding
        embedding_service = EmbeddingService()
        query_embedding = await embedding_service.generate_embedding(query.question)
        
        # Step 2: Search vector store
        vector_store = get_vector_store()
        top_k = query.top_k or settings.top_k_results
        
        search_results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Step 3: Retrieve chunk details from database
        chunk_ids = [result[0] for result in search_results]
        chunks = db.query(DocumentChunk)\
            .filter(DocumentChunk.id.in_(chunk_ids))\
            .all()
        
        # Create mapping for easy lookup
        chunk_map = {chunk.id: chunk for chunk in chunks}
        
        # Build context and retrieved chunks list
        contexts = []
        retrieved_chunks = []
        
        for chunk_id, similarity, metadata in search_results:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                contexts.append(chunk.text)
                
                retrieved_chunks.append(RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    similarity_score=round(similarity, 4),
                    chunk_index=chunk.chunk_index
                ))
        
        # Step 4: Generate answer using RAG
        rag_service = RAGService()
        answer = await rag_service.generate_answer(
            question=query.question,
            contexts=contexts
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResponse(
            answer=answer,
            question=query.question,
            chunks_used=retrieved_chunks,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )