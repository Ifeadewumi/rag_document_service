from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.database import get_db
from app.config import get_settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.schemas import (
    DocumentUploadResponse,
    DocumentListItem,
    DocumentDetailResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document (PDF, DOCX, or TXT).
    Extracts text, chunks it, generates embeddings, and stores in vector DB.
    """
    # Validate file type
    allowed_types = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt"
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: PDF, DOCX, TXT"
        )
    
    file_type = allowed_types[file.content_type]
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size allowed is {settings.max_upload_size_bytes / (1024*1024)}MB"
            )
        
        # Create document record
        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            status=DocumentStatus.PROCESSING
        )
        db.add(document)
        db.commit()
        
        # Process document
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Extract text
        extracted_text = processor.extract_text(file_content, file_type)
        document.extracted_text = extracted_text
        
        # Chunk text
        chunks = processor.chunk_text(extracted_text)
        
        if not chunks:
            document.status = DocumentStatus.FAILED
            document.error_message = "No text could be extracted from document"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted"
            )
        
        # Generate embeddings
        embedding_service = EmbeddingService()
        chunk_texts = [chunk[0] for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings(chunk_texts)
        
        # Store chunks in database and vector store
        vector_store = get_vector_store()
        chunk_ids = []
        chunk_metadatas = []
        
        for idx, (chunk_text, token_count) in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            
            # Save to database
            chunk_record = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=idx,
                text=chunk_text,
                token_count=token_count,
                vector_id=chunk_id
            )
            db.add(chunk_record)
            
            chunk_ids.append(chunk_id)
            chunk_metadatas.append({
                "document_id": doc_id,
                "chunk_index": idx,
                "filename": file.filename
            })
        
        # Add to vector store
        await vector_store.add_vectors(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            texts=chunk_texts
        )
        
        # Update document status
        document.chunk_count = len(chunks)
        document.status = DocumentStatus.COMPLETED
        db.commit()
        
        return DocumentUploadResponse(
            id=doc_id,
            filename=file.filename,
            file_type=file_type,
            status=DocumentStatus.COMPLETED,
            message=f"Document processed successfully. Created {len(chunks)} chunks."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Mark document as failed
        if document:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("", response_model=List[DocumentListItem])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all uploaded documents with their metadata and chunk counts.
    """
    documents = db.query(Document)\
        .order_by(Document.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return documents


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document_detail(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific document including all chunks.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get all chunks
    chunks = db.query(DocumentChunk)\
        .filter(DocumentChunk.document_id == document_id)\
        .order_by(DocumentChunk.chunk_index)\
        .all()
    
    chunks_data = [
        {
            "id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "token_count": chunk.token_count
        }
        for chunk in chunks
    ]
    
    return DocumentDetailResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        chunk_count=document.chunk_count,
        extracted_text=document.extracted_text,
        created_at=document.created_at,
        chunks=chunks_data
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all its chunks from both database and vector store.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Delete from vector store
        vector_store = get_vector_store()
        await vector_store.delete_by_document(document_id)
        
        # Delete chunks from database
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Delete document
        db.delete(document)
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )