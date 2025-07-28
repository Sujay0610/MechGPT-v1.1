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

class KnowledgeBaseService:
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection_name = "mechagent_knowledge_base"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "MechAgent RAG Knowledge Base"}
            )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Knowledge base initialized with collection: {self.collection_name}")
    
    async def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add text chunks to the knowledge base with embeddings
        """
        if not chunks:
            return 0
        
        try:
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
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            print(f"Added {len(texts)} chunks to knowledge base")
            return len(texts)
            
        except Exception as e:
            print(f"Error adding chunks to knowledge base: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant chunks
        """
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, 
                [query], 
                convert_to_tensor=False
            )
            
            # Search ChromaDB
            results = self.collection.query(
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
            print(f"Error searching knowledge base: {e}")
            return []
    
    async def reindex_all(self) -> int:
        """
        Reindex all parsed content from the parsed directory
        """
        try:
            # Clear existing collection
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "MechAgent RAG Knowledge Base"}
            )
            
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
            # Get collection count
            count = self.collection.count()
            
            # Get unique files
            all_metadata = self.collection.get(include=["metadatas"])
            unique_files = set()
            
            if all_metadata['metadatas']:
                for metadata in all_metadata['metadatas']:
                    if 'filename' in metadata:
                        unique_files.add(metadata['filename'])
            
            return {
                "total_chunks": count,
                "total_files": len(unique_files),
                "collection_name": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2",
                "last_updated": datetime.now().isoformat()
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
        Clean metadata to ensure it's JSON serializable for ChromaDB
        """
        clean_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, datetime):
                clean_metadata[key] = value.isoformat()
            else:
                clean_metadata[key] = str(value)
        
        return clean_metadata
    
    async def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific filename
        """
        try:
            # Get all items with the specified filename
            results = self.collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            print(f"Error deleting chunks for {filename}: {e}")
            return 0