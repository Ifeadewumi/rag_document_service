import httpx
from typing import List, Dict
from app.config import get_settings

settings = get_settings()


class RAGService:
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.model = settings.llm_model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def build_rag_prompt(self, question: str, contexts: List[str]) -> str:
        """Build RAG prompt with context."""
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents.

Context Documents:
{context_text}

User Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite which context(s) you used if relevant

Answer:"""
        
        return prompt
    
    async def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer using LLM."""
        prompt = self.build_rag_prompt(question, contexts)
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            
            return answer