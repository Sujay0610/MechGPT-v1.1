import os
import asyncio
from typing import List, Dict, Any
from llama_cloud_services import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from datetime import datetime
import uuid
from pathlib import Path

class PDFParserService:
    def __init__(self):
        self.api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        if not self.api_key:
            print("Warning: LLAMA_CLOUD_API_KEY not found in environment variables")
            self.parser = None
        else:
            # Configure LlamaParse for balanced mode (default)
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",  # Markdown preserves structure better for RAG
                # Using default balanced mode - best for documents with tables and images
                parse_mode="parse_page_with_llm",  # Balanced only
 # Optimized for technical docs with schematics
                system_prompt="""Extract all content with maximum fidelity including:
                - All text content with proper formatting
                - Tables with structure preserved
                - Technical diagrams and schematics with detailed descriptions
                - Headers, subheaders, and document structure
                - Lists, bullet points, and numbered items
                - Mathematical equations and formulas
                - Image captions and descriptions
                Maintain document hierarchy and relationships between sections.""",
                max_timeout=5000,  # Increased timeout for processing
                verbose=True,
                take_screenshot=True  # Capture page screenshots for reference
            )
        
        # Initialize text splitter optimized for markdown content
        self.text_splitter = SentenceSplitter(
            chunk_size=1500,  # Larger chunks for better context in RAG
            chunk_overlap=300,  # More overlap to preserve relationships
            separator="\n\n",  # Split on paragraph breaks for markdown
            paragraph_separator="\n\n\n",  # Preserve section breaks
            secondary_chunking_regex="[.!?]\s+"  # Fallback to sentence boundaries
        )
    
    async def parse_pdf(self, file_path: str, original_filename: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF file using LlamaParse and return chunks with proper async context management
        """
        if not self.parser:
            raise Exception("LlamaParse not configured. Please set LLAMA_CLOUD_API_KEY environment variable.")
        
        try:
            # Parse the PDF using LlamaParse async method
            result = await self.parser.aparse(file_path)
            
            if not result:
                raise Exception("No content extracted from PDF")
            
            # Get markdown documents from the result
            documents = result.get_markdown_documents(split_by_page=True)
            
            if not documents:
                raise Exception("No markdown documents extracted from PDF")
            
            chunks = []
            
            for doc_idx, document in enumerate(documents):
                # Split document into chunks
                nodes = self.text_splitter.get_nodes_from_documents([document])
                
                for node_idx, node in enumerate(nodes):
                    chunk_id = str(uuid.uuid4())
                    
                    # Analyze chunk content for better metadata
                    chunk_text = node.text.strip()
                    content_type = self._analyze_content_type(chunk_text)
                    
                    # Create enhanced chunk metadata for better RAG
                    metadata = {
                        "filename": original_filename,
                        "file_path": file_path,
                        "source": "pdf_upload_balanced",
                        "upload_time": datetime.now().isoformat(),
                        "file_size": os.path.getsize(file_path),
                        "document_index": doc_idx,
                        "chunk_index": node_idx,
                        "chunk_id": chunk_id,
                        "total_chunks": len(nodes),
                        "content_type": content_type,
                        "chunk_length": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "has_tables": "|" in chunk_text,
                        "has_headers": any(line.startswith("#") for line in chunk_text.split("\n")),
                        "has_lists": any(line.strip().startswith(("*", "-", "1.", "2.", "3.")) for line in chunk_text.split("\n")),
                        "has_code": "```" in chunk_text,
                        "section_level": self._get_section_level(chunk_text)
                    }
                    
                    # Add any existing metadata from the document
                    if hasattr(document, 'metadata') and document.metadata:
                        metadata.update(document.metadata)
                    
                    # Add any node-specific metadata
                    if hasattr(node, 'metadata') and node.metadata:
                        metadata.update(node.metadata)
                    
                    chunk = {
                        "text": node.text.strip(),
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    }
                    
                    # Only add non-empty chunks
                    if chunk["text"]:
                        chunks.append(chunk)
            
            print(f"Successfully parsed {original_filename}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            print(f"Error parsing PDF {original_filename}: {str(e)}")
            raise Exception(f"Failed to parse PDF: {str(e)}")
    
    def _analyze_content_type(self, text: str) -> str:
        """
        Analyze the content type of a text chunk for better RAG categorization
        """
        text_lower = text.lower()
        
        if "```" in text or "def " in text or "class " in text:
            return "code"
        elif "|" in text and "---" in text:
            return "table"
        elif any(line.startswith("#") for line in text.split("\n")):
            return "header"
        elif any(text_lower.startswith(keyword) for keyword in ["figure", "diagram", "image", "chart"]):
            return "figure"
        elif any(line.strip().startswith(("*", "-", "1.", "2.", "3.")) for line in text.split("\n")):
            return "list"
        elif len(text.split()) < 20:
            return "title_or_caption"
        else:
            return "paragraph"
    
    def _get_section_level(self, text: str) -> int:
        """
        Determine the section level based on markdown headers
        """
        lines = text.split("\n")
        for line in lines:
            if line.startswith("#"):
                return len(line) - len(line.lstrip("#"))
        return 0
    
    def extract_text_simple(self, file_path: str, original_filename: str) -> List[Dict[str, Any]]:
        """
        Fallback method for text extraction without LlamaParse
        """
        try:
            # This is a simple fallback - in a real implementation,
            # you might use PyPDF2 or pdfplumber here
            chunks = []
            
            # Create a single chunk with placeholder text
            chunk_id = str(uuid.uuid4())
            metadata = {
                "filename": original_filename,
                "file_path": file_path,
                "source": "pdf_upload_fallback",
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "chunk_id": chunk_id,
                "note": "Extracted without LlamaParse - limited functionality"
            }
            
            chunk = {
                "text": f"PDF content from {original_filename} (LlamaParse not available)",
                "metadata": metadata,
                "chunk_id": chunk_id
            }
            
            chunks.append(chunk)
            return chunks
            
        except Exception as e:
            raise Exception(f"Failed to extract text: {str(e)}")
    
    async def get_parser_status(self) -> Dict[str, Any]:
        """
        Get the status of the PDF parser service
        """
        return {
            "service": "PDFParserService",
            "llamaparse_configured": self.parser is not None,
            "api_key_present": bool(self.api_key),
            "status": "active" if self.parser else "limited"
        }