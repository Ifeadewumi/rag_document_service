import PyPDF2
from docx import Document as DocxDocument
import tiktoken
from typing import List, Tuple
import io


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """Extract text from different file types."""
        if file_type == "pdf":
            return self._extract_pdf(file_content)
        elif file_type == "docx":
            return self._extract_docx(file_content)
        elif file_type == "txt":
            return file_content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF."""
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    def _extract_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX."""
        docx_file = io.BytesIO(file_content)
        doc = DocxDocument(docx_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into chunks with overlap.
        Returns list of (chunk_text, token_count) tuples.
        """
        # Split into sentences (simple approach)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > self.chunk_size:
                # Add current chunk if it has content
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, current_tokens))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens > self.chunk_size:
                        if temp_chunk:
                            chunk_text = " ".join(temp_chunk)
                            chunks.append((chunk_text, temp_tokens))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
            
            # Check if adding sentence exceeds chunk size
            elif current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, current_tokens))
                
                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_tokens))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Split on common sentence endings
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences