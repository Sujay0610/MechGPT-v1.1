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
        """Get the status of the PDF parser service"""
        return {
            "service": "PDFParserService",
            "status": "active" if self.parser else "inactive",
            "api_key_configured": bool(self.api_key),
            "parser_configured": bool(self.parser),
            "timestamp": datetime.now().isoformat()
        }
    
    async def parse_text(self, content: str, title: str = "Text Content") -> List[Dict[str, Any]]:
        """Parse text content and return chunks"""
        try:
            # Create a document from the text content
            document = Document(text=content, metadata={"title": title, "source": "text_input"})
            
            # Split the text into chunks
            nodes = self.text_splitter.get_nodes_from_documents([document])
            
            chunks = []
            for i, node in enumerate(nodes):
                chunk_id = str(uuid.uuid4())
                chunk = {
                    "text": node.text.strip(),
                    "metadata": {
                        "source": title,
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "content_type": "text",
                        "created_at": datetime.now().isoformat(),
                        "title": title,
                        "filename": f"{title}.txt",
                        "chunk_id": chunk_id
                    },
                    "chunk_id": chunk_id
                }
                if chunk["text"]:  # Only add non-empty chunks
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error parsing text content: {e}")
            return []
    
    async def parse_url(self, url: str) -> List[Dict[str, Any]]:
        """Crawl a URL and parse its content into chunks"""
        try:
            # Try crawl4ai first, fallback to requests if it fails
            try:
                from crawl4ai import AsyncWebCrawler
                
                # Use simple configuration for Windows compatibility
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=url)
                    
                    if result.success and (result.markdown or result.cleaned_html):
                        title = getattr(result, 'title', None) or url.split('/')[-1] or "Web Page"
                        content = result.markdown or result.cleaned_html
                        
                        # Create a document from the crawled content
                        document = Document(
                            text=content, 
                            metadata={
                                "title": title, 
                                "source": url,
                                "url": url
                            }
                        )
                        
                        # Split the content into chunks
                        nodes = self.text_splitter.get_nodes_from_documents([document])
                        
                        chunks = []
                        for i, node in enumerate(nodes):
                            chunk_id = str(uuid.uuid4())
                            chunk = {
                                "text": node.text.strip(),
                                "metadata": {
                                    "source": title,
                                    "url": url,
                                    "chunk_index": i,
                                    "total_chunks": len(nodes),
                                    "content_type": "web_page",
                                    "created_at": datetime.now().isoformat(),
                                    "title": title,
                                    "filename": f"{title}.html",
                                    "chunk_id": chunk_id
                                },
                                "chunk_id": chunk_id
                            }
                            if chunk["text"]:  # Only add non-empty chunks
                                chunks.append(chunk)
                        
                        return chunks
                    else:
                        print(f"Crawl4ai failed for {url}, trying fallback method")
                        raise Exception("Crawl4ai failed")
                        
            except Exception as crawl4ai_error:
                print(f"Crawl4ai error: {crawl4ai_error}. Using fallback method...")
                
                # Fallback to simple requests + BeautifulSoup
                import requests
                from bs4 import BeautifulSoup
                import re
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else url.split('/')[-1] or "Web Page"
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in content.splitlines())
                chunks_text = '\n'.join(chunk for chunk in lines if chunk)
                
                if not chunks_text:
                    print(f"No content extracted from {url}")
                    return []
                
                # Create a document from the extracted content
                document = Document(
                    text=chunks_text, 
                    metadata={
                        "title": title, 
                        "source": url,
                        "url": url
                    }
                )
                
                # Split the content into chunks
                nodes = self.text_splitter.get_nodes_from_documents([document])
                
                chunks = []
                for i, node in enumerate(nodes):
                    chunk_id = str(uuid.uuid4())
                    chunk = {
                        "text": node.text.strip(),
                        "metadata": {
                            "source": title,
                            "url": url,
                            "chunk_index": i,
                            "total_chunks": len(nodes),
                            "content_type": "web_page",
                            "created_at": datetime.now().isoformat(),
                            "title": title,
                            "filename": f"{title}.html",
                            "chunk_id": chunk_id,
                            "extraction_method": "fallback_requests"
                        },
                        "chunk_id": chunk_id
                    }
                    if chunk["text"]:  # Only add non-empty chunks
                        chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            print(f"Error crawling URL {url}: {e}")
            return []