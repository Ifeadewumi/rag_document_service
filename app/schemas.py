from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
from app.models import DocumentStatus


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    status: DocumentStatus
    message: str


class DocumentListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: DocumentStatus
    chunk_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentDetailResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: DocumentStatus
    chunk_count: int
    extracted_text: Optional[str]
    created_at: datetime
    chunks: List[dict]
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    question: str
    chunks_used: List[RetrievedChunk]
    processing_time_ms: float