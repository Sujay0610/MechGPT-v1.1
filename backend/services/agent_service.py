import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
import time
from config.supabase_client import get_supabase_client

class AgentService:
    def __init__(self):
        # Initialize Supabase client
        self.supabase = get_supabase_client()
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Index configuration
        self.base_index_name = "mechagent-agents"
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Agent service initialized with Pinecone")
    
    def _get_agent_namespace(self, agent_name: str, user_id: str = None) -> str:
        """Generate a unique namespace for an agent"""
        if user_id:
            return f"user_{user_id}_agent_{agent_name}"
        return f"agent_{agent_name}"
    
    def _setup_index(self) -> bool:
        """Setup Pinecone index if it doesn't exist"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.base_index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.base_index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            return True
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            return False
    
    def _get_agents_for_user(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all agents for a specific user"""
        try:
            if user_id:
                result = self.supabase.table("agents").select("*").eq("user_id", user_id).execute()
            else:
                result = self.supabase.table("agents").select("*").execute()
            
            agents_data = result.data if result.data else []
            print(f"Raw agents data from Supabase: {agents_data}")
            return agents_data
        except Exception as e:
            print(f"Error loading agents: {e}")
            return []
    
    async def create_agent(self, name: str, description: str = "", extra_instructions: str = "", user_id: str = None) -> Dict[str, Any]:
        """Create a new agent"""
        try:
            # Check if agent already exists for this user
            existing_agent = self.supabase.table("agents").select("*").eq("name", name).eq("user_id", user_id).execute()
            if existing_agent.data:
                return {
                    "success": False,
                    "message": f"Agent '{name}' already exists"
                }
            
            # Setup Pinecone index if needed
            if not self._setup_index():
                return {
                    "success": False,
                    "message": "Failed to setup Pinecone index"
                }
            
            # Generate namespace for this agent
            namespace = self._get_agent_namespace(name, user_id)
            
            # Create agent metadata
            agent_id = str(uuid.uuid4())
            agent_data = {
                "id": agent_id,
                "name": name,
                "description": description,
                "extra_instructions": extra_instructions,
                "user_id": user_id,
                "collection_name": namespace,
                "total_files": 0,
                "total_chunks": 0
            }
            
            # Save agent to Supabase
            result = self.supabase.table("agents").insert(agent_data).execute()
            
            if result.data:
                return {
                    "success": True,
                    "message": f"Agent '{name}' created successfully",
                    "agent": result.data[0]
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to create agent in database"
                }
                
        except Exception as e:
            print(f"Error creating agent: {e}")
            return {
                "success": False,
                "message": f"Failed to create agent: {str(e)}"
            }
    
    async def get_agents(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all agents for a user"""
        try:
            raw_agents = self._get_agents_for_user(user_id)
            print(f"Processing {len(raw_agents)} agents")
            
            # Map database fields to expected schema fields
            mapped_agents = []
            for i, agent in enumerate(raw_agents):
                print(f"Processing agent {i}: {agent.keys() if isinstance(agent, dict) else type(agent)}")
                mapped_agent = {
                    "id": agent.get("id", ""),
                    "name": agent.get("name", ""),
                    "description": agent.get("description", ""),
                    "extra_instructions": agent.get("extra_instructions", ""),
                    "collection_name": agent.get("collection_name", ""),  # Use collection_name directly
                    "created_at": agent.get("created_at", ""),
                    "updated_at": agent.get("updated_at", ""),
                    "total_chunks": agent.get("total_chunks", 0),  # Use total_chunks directly
                    "total_files": agent.get("total_files", 0),  # Use total_files directly
                    "files": agent.get("files", [])  # Default to empty list if not present
                }
                mapped_agents.append(mapped_agent)
            return mapped_agents
        except Exception as e:
            print(f"Error getting agents: {e}")
            raise e
    
    async def get_agent(self, name: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific agent by name for a user"""
        try:
            query = self.supabase.table("agents").select("*").eq("name", name)
            if user_id is not None:
                query = query.eq("user_id", user_id)
            result = query.execute()
            if result.data:
                agent = result.data[0]
                # Map database fields to expected schema fields
                mapped_agent = {
                    "id": agent.get("id", ""),
                    "name": agent.get("name", ""),
                    "description": agent.get("description", ""),
                    "extra_instructions": agent.get("extra_instructions", ""),
                    "collection_name": agent.get("collection_name", ""),  # Use collection_name directly
                    "created_at": agent.get("created_at", ""),
                    "updated_at": agent.get("updated_at", ""),
                    "total_chunks": agent.get("total_chunks", 0),  # Use total_chunks directly
                    "total_files": agent.get("total_files", 0),  # Use total_files directly
                    "files": agent.get("files", [])  # Default to empty list if not present
                }
                return mapped_agent
            return None
        except Exception as e:
            print(f"Error getting agent: {e}")
            return None
    
    async def delete_agent(self, name: str, user_id: str = None) -> bool:
        """Delete an agent and its knowledge base"""
        try:
            # Get agent data first
            agent = await self.get_agent(name, user_id)
            if not agent:
                return False
            
            namespace = agent.get('collection_name')  # Use collection_name which stores the namespace
            
            # Delete from Pinecone namespace
            if namespace:
                try:
                    index = self.pc.Index(self.base_index_name)
                    # Delete all vectors in the namespace
                    index.delete(delete_all=True, namespace=namespace)
                    print(f"Deleted Pinecone namespace: {namespace}")
                except Exception as e:
                    print(f"Error deleting Pinecone namespace: {e}")
            
            # Delete all conversations for this agent
            try:
                from services.conversation_service import ConversationService
                conversation_service = ConversationService()
                deleted_conversations = await conversation_service.delete_agent_conversations(name, user_id)
                print(f"Deleted {deleted_conversations} conversations for agent '{name}'")
            except Exception as e:
                print(f"Error deleting agent conversations: {e}")
            
            # Delete from Supabase
            query = self.supabase.table("agents").delete().eq("name", name)
            if user_id is not None:
                query = query.eq("user_id", user_id)
            result = query.execute()
            
            if result.data is not None:  # Supabase returns empty list for successful deletes
                print(f"Deleted agent: {name}")
                return True
            else:
                return False
            
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
    
    async def add_chunks_to_agent(self, agent_name: str, chunks: List[Dict[str, Any]], filenames = None, user_id: str = None) -> int:
        """Add text chunks to a specific agent's knowledge base"""
        try:
            # Get agent data from Supabase
            agent = await self.get_agent(agent_name, user_id)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            namespace = agent.get('collection_name')
            if not namespace:
                raise ValueError(f"Agent '{agent_name}' has no collection_name configured")
            
            if not chunks:
                return 0
            
            # Handle filenames parameter - can be string, list, or None
            if isinstance(filenames, str):
                filenames = [filenames]
            elif filenames is None:
                filenames = []
            
            # Get Pinecone index
            index = self.pc.Index(self.base_index_name)
            
            # Prepare data for Pinecone
            vectors_to_upsert = []
            
            for chunk in chunks:
                text = chunk.get('text', '').strip()
                if not text:
                    continue
                
                chunk_id = chunk.get('chunk_id') or str(uuid.uuid4())
                metadata = chunk.get('metadata', {})
                
                # Add agent to metadata
                metadata['agent_name'] = agent_name
                metadata['user_id'] = user_id
                
                # Add filename to metadata if available
                if 'filename' in metadata:
                    # Use filename from chunk metadata if available
                    pass
                elif filenames:
                    # Use the first filename if available
                    metadata['filename'] = filenames[0]
                
                # Add text to metadata since Pinecone doesn't store documents separately
                metadata['text'] = text
                
                # Ensure metadata is JSON serializable
                clean_metadata = self._clean_metadata(metadata)
                
                # Generate embedding for this chunk
                embedding = await asyncio.to_thread(
                    self.embedding_model.encode, 
                    [text], 
                    convert_to_tensor=False
                )
                
                # Add to vectors list
                vectors_to_upsert.append({
                    'id': chunk_id,
                    'values': embedding[0].tolist(),
                    'metadata': clean_metadata
                })
            
            if not vectors_to_upsert:
                return 0
            
            # Upsert vectors to Pinecone
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            
            # Update agent metadata in Supabase
            supabase = get_supabase_client()
            current_chunks = agent.get('total_chunks', 0)
            current_files = agent.get('files', [])
            
            # Add new filenames to agent's files list
            if filenames:
                for filename in filenames:
                    if filename and filename not in current_files:
                        current_files.append(filename)
            
            # Update agent in Supabase
            update_data = {
                'total_chunks': current_chunks + len(vectors_to_upsert),
                'files': current_files,
                'total_files': len(current_files),
                'updated_at': datetime.now().isoformat()
            }
            
            supabase.table('agents').update(update_data).eq('name', agent_name).execute()
            
            print(f"Added {len(vectors_to_upsert)} chunks to agent '{agent_name}'")
            return len(vectors_to_upsert)
            
        except Exception as e:
            print(f"Error adding chunks to agent: {e}")
            raise
    
    async def search_agent(self, agent_name: str, query: str, top_k: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """Search a specific agent's knowledge base"""
        try:
            # Get agent data from Supabase
            agent = await self.get_agent(agent_name, user_id)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            namespace = agent.get('collection_name')
            if not namespace:
                raise ValueError(f"Agent '{agent_name}' has no collection_name configured")
            
            # Get Pinecone index
            index = self.pc.Index(self.base_index_name)
            
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, 
                [query], 
                convert_to_tensor=False
            )
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            # Format results
            search_results = []
            if results.get('matches'):
                for i, match in enumerate(results['matches']):
                    search_results.append({
                        "text": match.get('metadata', {}).get('text', ''),
                        "metadata": match.get('metadata', {}),
                        "similarity_score": match.get('score', 0),
                        "rank": i + 1
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching agent knowledge base: {e}")
            return []
    
    async def get_agent_stats(self, agent_name: str, user_id: str = None) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        try:
            # Get agent data from Supabase
            agent = await self.get_agent(agent_name, user_id)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            namespace = agent.get('collection_name')
            if not namespace:
                raise ValueError(f"Agent '{agent_name}' has no collection_name configured")
            
            # Get Pinecone index
            index = self.pc.Index(self.base_index_name)
            
            # Get current count from Pinecone
            stats = index.describe_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            count = namespace_stats.get('vector_count', 0)
            
            # Update agent metadata with current count in Supabase
            supabase = get_supabase_client()
            update_data = {
                'total_chunks': count,
                'updated_at': datetime.now().isoformat()
            }
            supabase.table('agents').update(update_data).eq('name', agent_name).execute()
            
            return {
                "agent_name": agent_name,
                "total_chunks": count,
                "total_files": len(agent.get("files", [])),
                "files": agent.get("files", []),
                "created_at": agent.get("created_at"),
                "updated_at": agent.get("updated_at"),
                "description": agent.get("description", ""),
                "extra_instructions": agent.get("extra_instructions", "")
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
    
    async def delete_file_from_agent(self, agent_name: str, filename: str, user_id: str = None) -> int:
        """Delete all chunks associated with a specific filename from an agent"""
        try:
            # Get agent data from Supabase
            agent = await self.get_agent(agent_name, user_id)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            namespace = agent.get('collection_name')
            if not namespace:
                raise ValueError(f"Agent '{agent_name}' has no collection_name configured")
            
            # Get Pinecone index
            index = self.pc.Index(self.base_index_name)
            
            # Query for vectors with the specified filename
            query_results = index.query(
                vector=[0] * 384,  # Dummy vector for metadata filtering
                top_k=10000,  # Large number to get all matches
                namespace=namespace,
                filter={"filename": filename},
                include_metadata=True
            )
            
            deleted_count = 0
            if query_results.get('matches'):
                # Extract IDs to delete
                ids_to_delete = [match['id'] for match in query_results['matches']]
                
                # Delete vectors from Pinecone
                index.delete(ids=ids_to_delete, namespace=namespace)
                deleted_count = len(ids_to_delete)
                
                # Update agent metadata in Supabase
                supabase = get_supabase_client()
                current_files = agent.get('files', [])
                current_chunks = agent.get('total_chunks', 0)
                
                if filename in current_files:
                    current_files.remove(filename)
                
                update_data = {
                    'files': current_files,
                    'total_files': len(current_files),
                    'total_chunks': max(0, current_chunks - deleted_count),
                    'updated_at': datetime.now().isoformat()
                }
                
                supabase.table('agents').update(update_data).eq('name', agent_name).execute()
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting file from agent: {e}")
            return 0
    
    async def reindex_agent_knowledge_base(self, agent_name: str, user_id: str = None) -> Dict[str, Any]:
        """Reindex an agent's knowledge base by clearing and rebuilding it"""
        try:
            # Get agent data from Supabase
            agent = await self.get_agent(agent_name, user_id)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            namespace = agent.get('collection_name')
            if not namespace:
                raise ValueError(f"Agent '{agent_name}' has no collection_name configured")
            
            # Get Pinecone index
            index = self.pc.Index(self.base_index_name)
            
            # Delete all vectors in the agent's namespace
            try:
                index.delete(delete_all=True, namespace=namespace)
                print(f"Cleared all vectors in namespace '{namespace}' for agent '{agent_name}'")
            except Exception as e:
                print(f"Warning: Could not clear namespace {namespace}: {e}")
            
            # Update agent metadata in Supabase to reset counts
            supabase = get_supabase_client()
            update_data = {
                'total_chunks': 0,
                'files': [],
                'total_files': 0,
                'updated_at': datetime.now().isoformat()
            }
            supabase.table('agents').update(update_data).eq('name', agent_name).execute()
            
            print(f"Successfully reindexed knowledge base for agent '{agent_name}'")
            
            return {
                "success": True,
                "message": f"Successfully reindexed knowledge base for agent '{agent_name}'",
                "namespace": namespace
            }
            
        except Exception as e:
            print(f"Error reindexing agent knowledge base: {e}")
            return {
                "success": False,
                "error": str(e)
            }