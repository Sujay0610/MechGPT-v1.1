import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

class AgentService:
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Agent metadata storage
        self.agents_file = Path("./data/agents.json")
        self.agents_file.parent.mkdir(exist_ok=True)
        
        print("Agent service initialized")
    
    def _load_agents(self) -> Dict[str, Any]:
        """Load agents metadata from file"""
        if self.agents_file.exists():
            try:
                with open(self.agents_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading agents: {e}")
        return {}
    
    def _save_agents(self, agents: Dict[str, Any]):
        """Save agents metadata to file"""
        try:
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(agents, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving agents: {e}")
    
    async def create_agent(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new agent with its own knowledge base"""
        try:
            agents = self._load_agents()
            
            # Check if agent already exists
            if name in agents:
                raise ValueError(f"Agent '{name}' already exists")
            
            # Create collection for this agent
            collection_name = f"agent_{name.lower().replace(' ', '_')}"
            
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Knowledge base for agent {name}"}
                )
            except Exception:
                # Collection might already exist
                collection = self.chroma_client.get_collection(collection_name)
            
            # Create agent metadata
            agent_data = {
                "id": str(uuid.uuid4()),
                "name": name,
                "description": description,
                "collection_name": collection_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total_chunks": 0,
                "total_files": 0,
                "files": []
            }
            
            agents[name] = agent_data
            self._save_agents(agents)
            
            print(f"Created agent '{name}' with collection '{collection_name}'")
            return agent_data
            
        except Exception as e:
            print(f"Error creating agent: {e}")
            raise
    
    async def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of all agents"""
        agents = self._load_agents()
        return list(agents.values())
    
    async def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific agent by name"""
        agents = self._load_agents()
        return agents.get(name)
    
    async def delete_agent(self, name: str) -> bool:
        """Delete an agent and its knowledge base"""
        try:
            agents = self._load_agents()
            
            if name not in agents:
                return False
            
            agent = agents[name]
            collection_name = agent["collection_name"]
            
            # Delete the collection
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception as e:
                print(f"Warning: Could not delete collection {collection_name}: {e}")
            
            # Remove from agents
            del agents[name]
            self._save_agents(agents)
            
            print(f"Deleted agent '{name}'")
            return True
            
        except Exception as e:
            print(f"Error deleting agent: {e}")
            return False
    
    async def is_file_already_processed(self, agent_name: str, filename: str) -> bool:
        """Check if a file has already been processed by an agent"""
        try:
            agents = self._load_agents()
            
            if agent_name not in agents:
                return False
            
            agent = agents[agent_name]
            return filename in agent.get("files", [])
            
        except Exception as e:
            print(f"Error checking if file is processed: {e}")
            return False
    
    async def add_chunks_to_agent(self, agent_name: str, chunks: List[Dict[str, Any]], filename: str = "") -> int:
        """Add text chunks to a specific agent's knowledge base"""
        try:
            agents = self._load_agents()
            
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            agent = agents[agent_name]
            collection_name = agent["collection_name"]
            
            # Get the collection
            collection = self.chroma_client.get_collection(collection_name)
            
            if not chunks:
                return 0
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                text = chunk.get('text', '').strip()
                if not text:
                    continue
                
                chunk_id = chunk.get('chunk_id') or str(uuid.uuid4())
                metadata = chunk.get('metadata', {})
                
                # Add agent and filename to metadata
                metadata['agent_name'] = agent_name
                if filename:
                    metadata['filename'] = filename
                
                # Ensure metadata is JSON serializable
                clean_metadata = self._clean_metadata(metadata)
                
                texts.append(text)
                metadatas.append(clean_metadata)
                ids.append(chunk_id)
            
            if not texts:
                return 0
            
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                self.embedding_model.encode, 
                texts, 
                convert_to_tensor=False
            )
            
            # Add to ChromaDB
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            # Update agent metadata
            agent["total_chunks"] += len(texts)
            agent["updated_at"] = datetime.now().isoformat()
            
            if filename and filename not in agent["files"]:
                agent["files"].append(filename)
                agent["total_files"] = len(agent["files"])
            
            agents[agent_name] = agent
            self._save_agents(agents)
            
            print(f"Added {len(texts)} chunks to agent '{agent_name}'")
            return len(texts)
            
        except Exception as e:
            print(f"Error adding chunks to agent: {e}")
            raise
    
    async def search_agent(self, agent_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search a specific agent's knowledge base"""
        try:
            agents = self._load_agents()
            
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            agent = agents[agent_name]
            collection_name = agent["collection_name"]
            
            # Get the collection
            collection = self.chroma_client.get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, 
                [query], 
                convert_to_tensor=False
            )
            
            # Search ChromaDB
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching agent knowledge base: {e}")
            return []
    
    async def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        try:
            agents = self._load_agents()
            
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            agent = agents[agent_name]
            collection_name = agent["collection_name"]
            
            # Get the collection
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get current count
            count = collection.count()
            
            # Update agent metadata with current count
            agent["total_chunks"] = count
            agents[agent_name] = agent
            self._save_agents(agents)
            
            return {
                "agent_name": agent_name,
                "total_chunks": count,
                "total_files": len(agent["files"]),
                "files": agent["files"],
                "created_at": agent["created_at"],
                "updated_at": agent["updated_at"],
                "description": agent["description"]
            }
            
        except Exception as e:
            print(f"Error getting agent stats: {e}")
            return {"error": str(e)}
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure it's JSON serializable for ChromaDB"""
        clean_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, datetime):
                clean_metadata[key] = value.isoformat()
            else:
                clean_metadata[key] = str(value)
        
        return clean_metadata
    
    async def delete_file_from_agent(self, agent_name: str, filename: str) -> int:
        """Delete all chunks associated with a specific filename from an agent"""
        try:
            agents = self._load_agents()
            
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            agent = agents[agent_name]
            collection_name = agent["collection_name"]
            
            # Get the collection
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get all items with the specified filename
            results = collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            deleted_count = 0
            if results['ids']:
                collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                
                # Update agent metadata
                if filename in agent["files"]:
                    agent["files"].remove(filename)
                    agent["total_files"] = len(agent["files"])
                
                agent["total_chunks"] -= deleted_count
                agent["updated_at"] = datetime.now().isoformat()
                
                agents[agent_name] = agent
                self._save_agents(agents)
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting file from agent: {e}")
            return 0