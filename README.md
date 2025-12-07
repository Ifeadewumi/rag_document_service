# RAG Document Service

A powerful **Retrieval-Augmented Generation (RAG)** backend service built with **FastAPI**. This application allows users to upload documents (PDF, DOCX, TXT), automatically indexes their content using vector embeddings, and enables AI-powered Question & Answering based specifically on those documents.

## 🚀 Features

- **Document Ingestion**: Seamlessly upload and process `.pdf`, `.docx`, and `.txt` files.
- **Smart Chunking**: Automatically extracts text and splits it into semantic chunks with overlap for better context preservation.
- **Vector Search**: Uses **ChromaDB** (local vector database) to store and retrieve relevant processed text chunks.
- **AI-Powered QA**: Integrates with **OpenRouter** (accessing models like Claude 3.5 Sonnet or GPT-4) to generate accurate answers based *only* on the retrieved context.
- **Modern Tech Stack**: Built with FastAPI, SQLAlchemy (PostgreSQL), and Pydantic.

## 🛠️ Architecture

1.  **Upload**: User uploads a file.
2.  **Process**: Service parses text and splits it into chunks.
3.  **Embed**: Chunks are converted to vector embeddings (currently via OpenRouter/OpenAI Compatible API).
4.  **Store**: Vectors are saved in ChromaDB; Metadata is saved in PostgreSQL.
5.  **Query**: User asks a question -> Question is embedded -> System searches ChromaDB for nearest vectors -> Top chunks + Question are sent to LLM -> LLM returns answer.

## 📋 Prerequisites

- **Python 3.10+**
- **PostgreSQL** (running locally or remote)
- **OpenRouter API Key** (for LLM and Embeddings)

## 🏁 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ifeadewumi/rag_document_service.git
cd rag_document_service
```

### 2. Set up Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory (based on the variables in `app/config.py`):

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db

# AI / OpenRouter
OPENROUTER_API_KEY=your_key_here
EMBEDDING_MODEL=openai/text-embedding-3-small
LLM_MODEL=anthropic/claude-3.5-sonnet

# Vector DB
VECTOR_DB_TYPE=chroma
CHROMA_PATH=./chroma_db
```

### 5. Run the Application
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

## 📚 API Documentation

Once running, visit **`http://localhost:8000/docs`** for the interactive Swagger UI.

### Key Endpoints

- **`POST /documents/upload`**: Upload a new file.
- **`GET /documents`**: List all uploaded documents and their status.
- **`POST /query`**: Ask a question about your documents.
  - Body: `{"question": "What does the document say about X?"}`

## 🗄️ Project Structure

```
rag_document_service/
├── app/
│   ├── routers/         # API Endpoints (documents.py, query.py)
│   ├── services/        # Business Logic (OCR, RAG, Vectors)
│   ├── models.py        # Database Models
│   ├── schemas.py       # Pydantic Schemas
│   └── main.py          # App Entrypoint
├── chroma_db/           # Local Vector Storage (Generated on runtime)
└── requirements.txt     # Python Dependencies
```
