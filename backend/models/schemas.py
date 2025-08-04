from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    chunks_found: int = 0
    conversation_id: Optional[str] = None
    
class ProcessedFile(BaseModel):
    filename: str
    file_size: int
    
class UploadResponse(BaseModel):
    message: str
    files: List[ProcessedFile]
    total_chunks: int
    
class TextChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
class SearchResult(BaseModel):
    chunk: TextChunk
    similarity_score: float
    source_file: str
    
class KnowledgeBaseStats(BaseModel):
    total_chunks: int
    total_files: int
    last_updated: Optional[datetime] = None
    
class CrawlRequest(BaseModel):
    urls: List[str]
    max_depth: int = 2
    include_pdfs: bool = True
    
class CrawlResponse(BaseModel):
    message: str
    crawled_urls: List[str]
    extracted_pdfs: List[str]
    total_chunks: int

# Agent-related schemas
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    extra_instructions: Optional[str] = ""

class Agent(BaseModel):
    id: str
    name: str
    description: str
    extra_instructions: str
    collection_name: str
    created_at: str
    updated_at: str
    total_chunks: int
    total_files: int
    files: List[str]

class AgentStats(BaseModel):
    agent_name: str
    total_chunks: int
    total_files: int
    files: List[str]
    created_at: str
    updated_at: str
    description: str
    extra_instructions: str

class AgentUploadRequest(BaseModel):
    agent_name: str

class AgentChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# Conversation history schemas
class ConversationMessage(BaseModel):
    id: str
    text: str
    sender: str  # 'user' or 'bot'
    timestamp: str
    agent_name: str
    conversation_id: str

class Conversation(BaseModel):
    id: str
    agent_name: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class ConversationHistory(BaseModel):
    conversation: Conversation
    messages: List[ConversationMessage]