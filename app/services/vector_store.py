from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings

settings = get_settings()


class VectorStore(ABC):
    @abstractmethod
    async def add_vectors(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict],
        texts: List[str]
    ):
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        pass
    
    @abstractmethod
    async def delete_by_document(self, document_id: str):
        pass


class ChromaVectorStore(VectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_vectors(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict],
        texts: List[str]
    ):
        """Add vectors to ChromaDB."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results: (id, distance, metadata)
        search_results = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Convert distance to similarity (cosine distance to similarity)
            similarity = 1 - distance
            
            search_results.append((chunk_id, similarity, metadata))
        
        return search_results
    
    async def delete_by_document(self, document_id: str):
        """Delete all chunks for a document."""
        # Query to get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])


def get_vector_store() -> VectorStore:
    """Factory function to get appropriate vector store."""
    if settings.vector_db_type == "chroma":
        return ChromaVectorStore()
    # elif settings.vector_db_type == "pinecone":
    #     return PineconeVectorStore()
    else:
        raise ValueError(f"Unsupported vector DB: {settings.vector_db_type}")