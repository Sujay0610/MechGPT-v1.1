import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid
import time

class KnowledgeBaseService:
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize OpenAI
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Index configuration
        self.index_name = "mechagent-knowledge-base"
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.embedding_model = "text-embedding-ada-002"
        
        # Create or connect to index
        self._setup_index()
        
        print(f"Knowledge base initialized with Pinecone index: {self.index_name}")
    
    def _setup_index(self):
        """Setup Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add text chunks to the knowledge base with embeddings
        """
        if not chunks:
            return 0
        
        try:
            # Prepare data for Pinecone
            vectors_to_upsert = []
            
            for chunk in chunks:
                text = chunk.get('text', '').strip()
                if not text:
                    continue
                
                chunk_id = chunk.get('chunk_id') or str(uuid.uuid4())
                metadata = chunk.get('metadata', {})
                
                # Ensure metadata is compatible with Pinecone
                clean_metadata = self._clean_metadata(metadata)
                # Add the text to metadata for retrieval
                clean_metadata['text'] = text
                
                # Generate embedding using OpenAI
                embedding = await self._generate_embedding(text)
                
                vectors_to_upsert.append({
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': clean_metadata
                })
            
            if not vectors_to_upsert:
                return 0
            
            # Upsert vectors to Pinecone in batches
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
            
            print(f"Added {total_upserted} chunks to Pinecone knowledge base")
            return total_upserted
            
        except Exception as e:
            print(f"Error adding chunks to knowledge base: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant chunks
        """
        try:
            # Generate query embedding using OpenAI
            query_embedding = await self._generate_embedding(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            search_results = []
            for i, match in enumerate(results['matches']):
                metadata = match.get('metadata', {})
                text = metadata.pop('text', '')  # Remove text from metadata for clean response
                
                search_results.append({
                    "text": text,
                    "metadata": metadata,
                    "similarity_score": match['score'],
                    "rank": i + 1
                })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    async def reindex_all(self) -> int:
        """
        Reindex all parsed content from the parsed directory
        """
        try:
            # Delete all vectors in the index (clear existing data)
            # Pinecone doesn't have a direct "clear all" method, so we'll delete by namespace
            # or recreate the index if needed
            try:
                # Get all vector IDs and delete them
                stats = self.index.describe_index_stats()
                if stats['total_vector_count'] > 0:
                    # For simplicity, we'll delete and recreate the index
                    self.pc.delete_index(self.index_name)
                    self._setup_index()
            except Exception as e:
                print(f"Warning: Could not clear existing index: {e}")
            
            # Load all parsed files
            parsed_dir = Path("parsed")
            total_chunks = 0
            
            if parsed_dir.exists():
                for json_file in parsed_dir.glob("*_parsed.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        
                        added_count = await self.add_chunks(chunks)
                        total_chunks += added_count
                        
                    except Exception as e:
                        print(f"Error processing {json_file}: {e}")
                        continue
            
            print(f"Reindexed {total_chunks} chunks from parsed files")
            return total_chunks
            
        except Exception as e:
            print(f"Error during reindexing: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics
        """
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            # Get unique files by querying metadata
            # Note: This is a simplified approach. In production, you might want to
            # maintain a separate metadata store for more efficient stats
            unique_files = set()
            
            # For now, we'll estimate based on available stats
            # In a production setup, you might want to maintain file counts separately
            
            return {
                "total_chunks": stats.get('total_vector_count', 0),
                "total_files": "estimated",  # Pinecone doesn't provide easy file counting
                "index_name": self.index_name,
                "embedding_model": "all-MiniLM-L6-v2",
                "dimension": self.dimension,
                "last_updated": datetime.now().isoformat(),
                "index_stats": stats
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "total_files": 0,
                "error": str(e)
            }
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure it's compatible with Pinecone
        Pinecone has specific requirements for metadata:
        - Values must be strings, numbers, booleans, or lists of strings
        - Metadata size limit is 40KB per vector
        """
        clean_metadata = {}
        
        for key, value in metadata.items():
            # Pinecone metadata key restrictions
            clean_key = str(key).replace('.', '_').replace('$', '_')
            
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[clean_key] = value
            elif isinstance(value, datetime):
                clean_metadata[clean_key] = value.isoformat()
            elif isinstance(value, list):
                # Convert list to string representation
                clean_metadata[clean_key] = str(value)
            else:
                clean_metadata[clean_key] = str(value)
        
        return clean_metadata
    
    async def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific filename
        """
        try:
            # Query vectors with the specified filename
            # Note: Pinecone doesn't support direct metadata filtering for deletion
            # We need to query first, then delete by IDs
            
            # This is a simplified approach - in production you might want to
            # maintain a mapping of filenames to vector IDs
            
            # Query to find vectors with this filename
            # Since we can't directly filter by metadata in query, we'll use a dummy vector
            dummy_vector = [0.0] * self.dimension
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=10000,  # Large number to get all results
                include_metadata=True,
                filter={"filename": filename}
            )
            
            if results['matches']:
                ids_to_delete = [match['id'] for match in results['matches']]
                
                # Delete in batches
                batch_size = 1000
                total_deleted = 0
                
                for i in range(0, len(ids_to_delete), batch_size):
                    batch_ids = ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch_ids)
                    total_deleted += len(batch_ids)
                
                return total_deleted
            
            return 0
            
        except Exception as e:
            print(f"Error deleting chunks for {filename}: {e}")
            return 0