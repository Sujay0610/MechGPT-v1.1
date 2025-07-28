from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import asyncio
from datetime import datetime
import aiofiles
from pathlib import Path
from dotenv import load_dotenv
import uuid
import threading

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from services.pdf_parser import PDFParserService
from services.knowledge_base import KnowledgeBaseService
from services.chat_service import ChatService
from services.agent_service import AgentService
from services.conversation_service import ConversationService
from models.schemas import (
    ChatRequest, ChatResponse, UploadResponse,
    AgentCreate, Agent, AgentStats, AgentUploadRequest, AgentChatRequest,
    Conversation, ConversationHistory, ConversationMessage
)

app = FastAPI(
    title="MechAgent RAG Backend",
    description="FastAPI backend for RAG chatbot with LlamaParse integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_parser = PDFParserService()
knowledge_base = KnowledgeBaseService()
agent_service = AgentService()
conversation_service = ConversationService()

# Job tracking for background processing
processing_jobs = {}
job_lock = threading.Lock()
chat_service = ChatService(knowledge_base, agent_service)

# Background processing function
async def process_files_background(job_id: str, agent_name: str, files_data: List[dict]):
    """Process files in the background"""
    try:
        with job_lock:
            processing_jobs[job_id] = {
                "status": "processing",
                "progress": 0,
                "total_files": len(files_data),
                "processed_files": [],
                "skipped_files": [],
                "failed_files": [],
                "total_chunks": 0,
                "message": "Starting file processing...",
                "started_at": datetime.now().isoformat()
            }
        
        for i, file_data in enumerate(files_data):
            filename = file_data["filename"]
            file_path = file_data["file_path"]
            
            # Update progress
            with job_lock:
                processing_jobs[job_id]["progress"] = i
                processing_jobs[job_id]["message"] = f"Processing {filename}..."
            
            try:
                # Check if file is already processed
                if await agent_service.is_file_already_processed(agent_name, filename):
                    with job_lock:
                        processing_jobs[job_id]["skipped_files"].append({
                            "filename": filename,
                            "reason": "Already processed"
                        })
                    continue
                
                # Parse with LlamaParse
                chunks = await pdf_parser.parse_pdf(file_path, filename)
                
                if not chunks:
                    with job_lock:
                        processing_jobs[job_id]["failed_files"].append({
                            "filename": filename,
                            "reason": "No content extracted from PDF"
                        })
                    continue
                
                # Save parsed chunks
                parsed_file = f"parsed/{agent_name}_{Path(file_path).stem}_parsed.json"
                async with aiofiles.open(parsed_file, 'w') as f:
                    await f.write(json.dumps(chunks, indent=2))
                
                # Index chunks in agent's knowledge base
                added_chunks = await agent_service.add_chunks_to_agent(agent_name, chunks, filename)
                
                with job_lock:
                    processing_jobs[job_id]["processed_files"].append({
                        "filename": filename,
                        "file_size": file_data["file_size"]
                    })
                    processing_jobs[job_id]["total_chunks"] += len(chunks)
                
                print(f"Successfully processed {filename}: {len(chunks)} chunks created")
                print(f"Added {added_chunks} chunks to agent '{agent_name}'")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                with job_lock:
                    processing_jobs[job_id]["failed_files"].append({
                        "filename": filename,
                        "reason": f"Processing failed: {str(e)}"
                    })
        
        # Mark as completed
        with job_lock:
            job = processing_jobs[job_id]
            job["status"] = "completed"
            job["progress"] = len(files_data)
            job["completed_at"] = datetime.now().isoformat()
            
            # Create final message
            message_parts = []
            if job["processed_files"]:
                message_parts.append(f"Successfully processed {len(job['processed_files'])} new file(s)")
            if job["skipped_files"]:
                message_parts.append(f"Skipped {len(job['skipped_files'])} file(s)")
            if job["failed_files"]:
                message_parts.append(f"Failed to process {len(job['failed_files'])} file(s)")
            
            job["message"] = " and ".join(message_parts) + f" for agent '{agent_name}'."
            
            # Add failure details
            if job["failed_files"]:
                failure_details = []
                for failed in job["failed_files"]:
                    failure_details.append(f"- {failed['filename']}: {failed['reason']}")
                job["message"] += f"\n\nFailed files:\n" + "\n".join(failure_details)
            
            # Get final agent stats
            agent_stats = await agent_service.get_agent_stats(agent_name)
            job["final_total_chunks"] = agent_stats.get("total_chunks", 0)
    
    except Exception as e:
        print(f"Background processing error: {e}")
        with job_lock:
            processing_jobs[job_id]["status"] = "failed"
            processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

# Create necessary directories
Path("uploads").mkdir(exist_ok=True)
Path("parsed").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "MechAgent RAG Backend API", "status": "running"}

# Agent Management Endpoints
@app.post("/api/agents", response_model=Agent)
async def create_agent(agent_data: AgentCreate):
    """
    Create a new agent with its own knowledge base
    """
    try:
        agent = await agent_service.create_agent(agent_data.name, agent_data.description)
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents", response_model=List[Agent])
async def get_agents():
    """
    Get list of all agents
    """
    try:
        agents = await agent_service.get_agents()
        return agents
    except Exception as e:
        print(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_name}", response_model=Agent)
async def get_agent(agent_name: str):
    """
    Get specific agent by name
    """
    try:
        agent = await agent_service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/agents/{agent_name}")
async def delete_agent(agent_name: str):
    """
    Delete an agent and its knowledge base
    """
    try:
        success = await agent_service.delete_agent(agent_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        return {"message": f"Agent '{agent_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_name}/stats", response_model=AgentStats)
async def get_agent_stats(agent_name: str):
    """
    Get statistics for a specific agent
    """
    try:
        stats = await agent_service.get_agent_stats(agent_name)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/{agent_name}/upload")
async def upload_files_to_agent(agent_name: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload and process files for a specific agent (background processing)
    """
    try:
        # Check if agent exists
        agent = await agent_service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Prepare files data for background processing
        files_data = []
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                continue  # Skip non-PDF files
            
            # Save uploaded file
            file_path = f"uploads/{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            files_data.append({
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(content)
            })
        
        if not files_data:
            raise HTTPException(status_code=400, detail="No valid PDF files found")
        
        # Start background processing
        background_tasks.add_task(process_files_background, job_id, agent_name, files_data)
        
        return {
            "success": True,
            "job_id": job_id,
            "message": f"Started processing {len(files_data)} file(s) for agent '{agent_name}'. Use the job_id to check status.",
            "total_files": len(files_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/agents/{agent_name}/upload/status/{job_id}")
async def get_upload_status(agent_name: str, job_id: str):
    """
    Get the status of a background upload job
    """
    with job_lock:
        if job_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = processing_jobs[job_id].copy()
    
    return job

@app.post("/api/agents/{agent_name}/chat", response_model=ChatResponse)
async def chat_with_agent(agent_name: str, request: AgentChatRequest):
    """
    Chat with a specific agent using its knowledge base
    """
    try:
        # Check if agent exists
        agent = await agent_service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Handle conversation creation/continuation
        conversation_id = request.conversation_id
        if not conversation_id:
            # Create new conversation
            conversation_id = await conversation_service.create_conversation(agent_name, request.message)
        
        # Add user message to conversation
        await conversation_service.add_message(conversation_id, request.message, "user", agent_name)
        
        # Get response from chat service
        response = await chat_service.get_response(
            message=request.message,
            conversation_id=conversation_id,
            agent_id=agent_name
        )
        
        # Add bot response to conversation
        await conversation_service.add_message(conversation_id, response.response, "bot", agent_name)
        
        # Add conversation_id to response
        response.conversation_id = conversation_id
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload and parse PDF files using LlamaParse
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processed_files = []
        total_chunks = 0
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
                
            # Save uploaded file
            file_path = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Parse with LlamaParse
            try:
                chunks = await pdf_parser.parse_pdf(file_path, file.filename)
                
                # Save parsed chunks
                parsed_file = f"parsed/{Path(file_path).stem}_parsed.json"
                async with aiofiles.open(parsed_file, 'w') as f:
                    await f.write(json.dumps(chunks, indent=2))
                
                # Index chunks in knowledge base
                await knowledge_base.add_chunks(chunks)
                
                processed_files.append({
                    "original_name": file.filename,
                    "file_path": file_path,
                    "parsed_file": parsed_file
                })
                
                total_chunks += len(chunks)
                
            except Exception as parse_error:
                print(f"Error parsing {file.filename}: {parse_error}")
                continue
        
        if not processed_files:
            raise HTTPException(status_code=400, detail="No valid PDF files were processed")
        
        return UploadResponse(
            message=f"Successfully processed {len(processed_files)} file(s) and created {total_chunks} text chunks.",
            files=processed_files,
            total_chunks=total_chunks
        )
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat messages and return RAG-based responses
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get response from chat service
        response = await chat_service.get_response(request.message)
        
        return response
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/crawl")
async def crawl_website(urls: List[str] = Form(...), max_depth: int = Form(2)):
    """
    Crawl websites and extract content (Phase 2 functionality)
    """
    # Placeholder for Phase 2
    return {"message": "Website crawling will be implemented in Phase 2"}

@app.post("/api/index")
async def index_content():
    """
    Index parsed content into vector store (Phase 3 functionality)
    """
    try:
        # Re-index all parsed content
        indexed_count = await knowledge_base.reindex_all()
        return {
            "message": f"Successfully indexed {indexed_count} chunks",
            "indexed_chunks": indexed_count
        }
    except Exception as e:
        print(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """
    Get system status and statistics
    """
    try:
        stats = await knowledge_base.get_stats()
        return {
            "status": "running",
            "knowledge_base_stats": stats,
            "services": {
                "pdf_parser": "active",
                "knowledge_base": "active",
                "chat_service": "active"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Conversation Management Endpoints
@app.get("/api/agents/{agent_name}/conversations", response_model=List[Conversation])
async def get_agent_conversations(agent_name: str):
    """
    Get all conversations for a specific agent
    """
    try:
        # Check if agent exists
        agent = await agent_service.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        conversations = await conversation_service.get_agent_conversations(agent_name)
        return conversations
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """
    Get full conversation history including messages
    """
    try:
        history = await conversation_service.get_conversation_history(conversation_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages
    """
    try:
        success = await conversation_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
        
        return {"message": f"Conversation '{conversation_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)