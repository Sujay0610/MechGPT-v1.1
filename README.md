# MechAgent RAG Chatbot

A plug-and-play, SaaS-style RAG (Retrieval-Augmented Generation) chatbot prototype for product manuals and documentation.

## Features

- **PDF Upload & Parsing**: Upload product manuals (PDFs) and parse them using LlamaParse
- **Vector Search**: ChromaDB with Sentence Transformers for semantic search
- **Chat Interface**: Minimal Next.js chat UI for interacting with your knowledge base
- **FastAPI Backend**: Robust Python backend handling all AI/ML operations
- **Extensible Architecture**: Ready for website crawling, advanced embeddings, and LLM integration

## Architecture

- **Frontend**: Next.js 14, React, Tailwind CSS, TypeScript
- **Backend**: FastAPI (Python) with async support
- **PDF Parsing**: LlamaParse integration
- **Vector Database**: ChromaDB for embeddings storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **File Storage**: Local filesystem with organized structure

## Setup Instructions

### 1. Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your LlamaParse API key:
   ```
   LLAMA_CLOUD_API_KEY=your_api_key_here
   ```
   Get your API key from [LlamaIndex Cloud](https://cloud.llamaindex.ai/)

5. Start the FastAPI server:
   ```bash
   python run.py
   ```
   
   The backend will be available at [http://localhost:8000](http://localhost:8000)
   API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend Setup (Next.js)

1. In a new terminal, navigate to the project root:
   ```bash
   cd ..
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env.local
   ```

4. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   
   The frontend will be available at [http://localhost:3000](http://localhost:3000)

## Usage

### Phase 1: PDF Upload & Chat (Current)

1. **Upload Manuals**:
   - Go to `/upload`
   - Drag and drop PDF files or click to browse
   - Files will be parsed using LlamaParse and chunked for retrieval

2. **Chat with Your Knowledge Base**:
   - Go to `/` (home page)
   - Ask questions about your uploaded content
   - The system will search for relevant chunks and provide answers

## Project Structure

```
MechAgent/
├── app/                          # Next.js frontend
│   ├── api/
│   │   ├── chat/
│   │   │   └── route.ts          # Chat API proxy to FastAPI
│   │   └── upload/
│   │       └── route.ts          # Upload API proxy to FastAPI
│   ├── context/
│   │   └── ChatContext.tsx       # React context for chat state
│   ├── upload/
│   │   └── page.tsx              # PDF upload page
│   ├── globals.css               # Global styles
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Main chat page
├── backend/                      # FastAPI backend
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # LlamaParse integration
│   │   ├── knowledge_base.py     # ChromaDB vector storage
│   │   └── chat_service.py       # RAG chat logic
│   ├── uploads/                  # Uploaded PDF files
│   ├── parsed/                   # Parsed content chunks (JSON)
│   ├── data/                     # ChromaDB storage
│   ├── main.py                   # FastAPI application
│   ├── run.py                    # Server startup script
│   ├── requirements.txt          # Python dependencies
│   └── .env.example              # Backend environment template
├── package.json                  # Node.js dependencies
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── .env.example                  # Frontend environment template
└── README.md
```

## Upcoming Phases

### Phase 2: Website Link Ingestion
- Add URL input to upload page
- Integrate Crawl4AI for recursive crawling
- Extract and parse linked PDFs

### Phase 3: Vector Indexing
- Integrate ChromaDB for vector storage
- Add Sentence Transformers for embeddings
- Implement semantic search

### Phase 4: Advanced RAG
- Integrate OpenAI or other LLMs
- Implement proper RAG pipeline
- Add conversation memory and context

## Development Notes

- The current implementation uses simple text similarity for search
- LlamaParse requires an API key for PDF parsing
- Files are stored locally in `uploads/` and `parsed/` directories
- The chat interface is functional but uses placeholder responses

## Contributing

This is a prototype implementation. Future enhancements will include:
- Vector embeddings and semantic search
- LLM integration for response generation
- Website crawling capabilities
- Production-ready deployment options