import os
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from models.schemas import ChatResponse
from services.knowledge_base import KnowledgeBaseService
from services.agent_service import AgentService
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.schema import HumanMessage, SystemMessage

class ChatService:
    def __init__(self, knowledge_base: KnowledgeBaseService, agent_service: AgentService):
        self.knowledge_base = knowledge_base
        self.agent_service = agent_service
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        
        # Simple cache for recent queries (in production, use Redis)
        self._query_cache = {}
        self._cache_max_size = 100
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize web search tool
        self.web_search_tool = self._initialize_web_search()
        
        # System prompt for technical assistance
        self.system_prompt = self._get_system_prompt()
        
        print(f"ChatService initialized with LLM: {bool(self.llm)}, Web Search: {bool(self.web_search_tool)}")
    
    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize the LLM with proper error handling"""
        if not self.openrouter_api_key:
            print("Warning: OPENROUTER_API_KEY not found. LLM features disabled.")
            return None
        
        try:
            # Using Claude 3.5 Haiku for faster, more concise responses
            # Alternative options: "anthropic/claude-3-5-haiku", "openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct"
            return ChatOpenAI(
                model="openai/gpt-4o-mini",  # Fast, concise, good for chatbot responses
                temperature=0.2,  # Slightly higher for more natural responses
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1/",
                max_tokens=500  # Reduced for shorter responses
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return None
    
    def _initialize_web_search(self) -> Optional[GoogleSerperAPIWrapper]:
        """Initialize web search with proper error handling"""
        if not self.serper_api_key:
            print("Warning: SERPER_API_KEY not found. Web search disabled.")
            return None
        
        try:
            return GoogleSerperAPIWrapper(serper_api_key=self.serper_api_key)
        except Exception as e:
            print(f"Warning: Could not initialize web search tool: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the technical assistant"""
        return """You are a friendly and knowledgeable technical support chatbot helping maintenance technicians with equipment issues.

Your personality:
- Conversational and approachable, like a helpful colleague
- Professional but not overly formal
- Patient and understanding when users need clarification
- Enthusiastic about helping solve technical problems

Your expertise:
- Equipment maintenance, troubleshooting, and operations
- Safety procedures and best practices
- Step-by-step technical guidance
- Part identification and replacement procedures

Your communication style:
- Use friendly, conversational language
- Start with direct answers, then offer additional help
- Include relevant links when available to help users learn more
- Ask follow-up questions to better assist users
- Use emojis sparingly but appropriately (⚠️ for warnings, ✅ for confirmations)
- Always prioritize safety in your recommendations

Remember: You're here to make technical support feel less intimidating and more collaborative!"""
        
    async def _search_knowledge_base(self, query: str, agent_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base with caching for faster responses"""
        try:
            # Create cache key
            cache_key = f"{agent_id or 'global'}:{query.lower().strip()}:{top_k}"
            
            # Check cache first
            if cache_key in self._query_cache:
                print(f"Cache hit for query: {query[:50]}...")
                return self._query_cache[cache_key]
            
            # Perform search
            if agent_id:
                results = await self.agent_service.search_agent(agent_id, query, top_k=top_k)
            else:
                results = await self.knowledge_base.search(query, top_k=top_k)
            
            # Cache results (limit cache size)
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = results
            print(f"Knowledge base search: {len(results)} total results (cached)")
            return results
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    async def _search_web(self, query: str) -> Dict[str, Any]:
        """Search the web for additional information and extract links"""
        if not self.web_search_tool:
            return {"text": "", "links": []}
        
        try:
            print(f"Performing web search for: {query}")
            # Use results() method to get structured JSON data from Serper.dev
            raw_results = await asyncio.to_thread(self.web_search_tool.results, query)
            
            # Parse and extract structured data from search results
            parsed_results = self._parse_web_results(raw_results)
            return parsed_results
            
        except Exception as e:
            print(f"Web search error: {e}")
            return {"text": "", "links": []}
    
    def _parse_web_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse web search results to extract links and structured information"""
        links = []
        text_content = []
        
        try:
            # raw_results is already a dictionary from GoogleSerperAPIWrapper
            data = raw_results
            
            # Extract organic results
            if 'organic' in data:
                for i, result in enumerate(data['organic'][:5]):  # Top 5 results
                    title = result.get('title', '')
                    link = result.get('link', '')
                    snippet = result.get('snippet', '')
                    
                    if link and title:
                        links.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet[:200] + '...' if len(snippet) > 200 else snippet
                        })
                        
                        text_content.append(f"**{title}**\n{snippet}\nSource: {link}\n")
            
            # Extract answer box if available
            if 'answerBox' in data:
                answer = data['answerBox']
                if 'answer' in answer:
                    text_content.insert(0, f"**Quick Answer:** {answer['answer']}\n")
                if 'link' in answer:
                    links.insert(0, {
                        'title': 'Answer Source',
                        'url': answer['link'],
                        'snippet': answer.get('answer', '')[:200]
                    })
            
            # Extract knowledge graph if available
            if 'knowledgeGraph' in data:
                kg = data['knowledgeGraph']
                if 'title' in kg and 'description' in kg:
                    text_content.insert(0, f"**{kg['title']}**\n{kg['description']}\n")
                    if 'website' in kg:
                        links.insert(0, {
                            'title': kg['title'],
                            'url': kg['website'],
                            'snippet': kg.get('description', '')[:200]
                        })
        
        except Exception as e:
            print(f"Error parsing web results: {e}")
            # Fallback: try to extract any useful information
            if isinstance(raw_results, dict):
                text_content.append(str(raw_results)[:1000])
            else:
                text_content.append(str(raw_results)[:1000] if raw_results else "")
        
        return {
            "text": "\n\n".join(text_content),
            "links": links
        }
    
    def _should_perform_web_search(self, message: str, kb_context: str) -> bool:
        """Determine if web search should be performed based on context quality and user intent"""
        message_lower = message.lower()
        
        # Always search if user explicitly requests web/online information
        web_keywords = [
            'search online', 'search web', 'find online', 'look up online',
            'google', 'internet', 'website', 'url', 'link', 'online',
            'current', 'latest', 'recent', 'new', 'updated', 'today',
            'official website', 'manufacturer website', 'download',
            'buy', 'purchase', 'price', 'cost', 'where to buy'
        ]
        
        if any(keyword in message_lower for keyword in web_keywords):
            return True
        
        # Search if knowledge base results are insufficient
        if not kb_context or len(kb_context.strip()) < 100:
            return True
        
        # Search for specific product/model queries that might need official links
        product_patterns = [
            r'\b[A-Z]{2,}\d+[A-Z]*\b',  # Product codes like UR10e, ABC123
            r'\bmodel\s+\w+',  # "model XYZ"
            r'\bpart\s+number',  # "part number"
            r'\bserial\s+number',  # "serial number"
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                # Only search if we don't have comprehensive info
                if len(kb_context.strip()) < 300:
                    return True
        
        # Don't search for general troubleshooting if we have good context
        return False
    
    def _build_context(self, chunks: List[Dict[str, Any]], web_results: Dict[str, Any] = None) -> str:
        """Build context from knowledge base chunks and web results"""
        context_parts = []
        
        if chunks:
            # Add document context
            doc_context = "Technical Documentation:\n"
            for i, chunk in enumerate(chunks[:3], 1):  # Limit to top 3 chunks
                text = chunk.get('text', '').strip()
                filename = chunk.get('metadata', {}).get('filename', 'Unknown')
                doc_context += f"\n[Source {i}: {filename}]\n{text}\n"
            context_parts.append(doc_context)
        
        if web_results and web_results.get('text'):
            context_parts.append(f"Web Search Results:\n{web_results['text']}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> List:
        """Create a structured prompt for the LLM"""
        if context:
            user_prompt = f"""Based on the following technical documentation and information, please answer the user's question:

{context}

User Question: {query}

Provide a detailed, practical response based on the available information. If the context doesn't fully answer the question, mention what information is available and what might be missing."""
        else:
            user_prompt = f"""The user asked: {query}

I don't have specific technical documentation available for this question. Please provide a helpful response and suggest uploading relevant technical manuals or documentation."""
        
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
    async def get_response(self, message: str, conversation_id: Optional[str] = None, agent_id: Optional[str] = None) -> ChatResponse:
        """Get response using LangChain-style RAG approach similar to Streamlit example"""
        try:
            print(f"\n=== RAG Query Processing ===")
            print(f"Processing message: {message}")
            
            # Step 1: Try knowledge base search first
            chunks = await self._search_knowledge_base(message, agent_id)
            kb_context = self._build_context(chunks) if chunks else ""
            
            print(f"Knowledge base results: {len(kb_context)} characters")
            
            # Step 2: Determine if web search is needed
            web_results = None
            web_links = []
            should_search_web = self._should_perform_web_search(message, kb_context)
            
            if should_search_web and self.web_search_tool:
                print("Performing web search...")
                web_results = await self._search_web(message)
                web_links = web_results.get('links', [])
                print(f"Web search results: {len(web_results.get('text', ''))} characters, {len(web_links)} links")
            else:
                print("Skipping web search - sufficient knowledge base results or not needed")
            
            # Step 3: Combine results
            context = self._build_context(chunks, web_results)
            
            if not context:
                print("No context found for query")
                return ChatResponse(
                    response="I'm sorry, I couldn't find any relevant information for your query. Please upload relevant technical documentation or try rephrasing your question.",
                    sources=[],
                    chunks_found=0
                )
            
            # Step 4: Generate response using LLM if available
            if self.llm:
                try:
                    # Create optimized chatbot prompt - include links only if web search was performed
                    links_text = ""
                    link_guidance = ""
                    
                    if web_links and should_search_web:
                        links_text = "\n\nRELEVANT LINKS (include these when helpful):\n"
                        for i, link in enumerate(web_links[:3], 1):
                            links_text += f"{i}. [{link['title']}]({link['url']})\n"
                        link_guidance = "- When you have relevant links, include them naturally in your response\n"
                    
                    prompt = f"""You are a friendly technical support chatbot helping maintenance technicians. Provide helpful, conversational answers based on the information below.

Technical Documentation:
{context}{links_text}

User Question: {message}

CHATBOT RESPONSE GUIDELINES:
- Be conversational and helpful, like talking to a colleague
- Keep initial answers concise (2-3 sentences) but offer to elaborate
- Use bullet points for step-by-step instructions
- Include part numbers and safety warnings when available
{link_guidance}- If info is incomplete, suggest specific next steps or resources
- Use friendly language: "Here's what I found...", "You'll want to...", "Let me help with that..."

EXAMPLE RESPONSES:
Q: "How do I reset the system?"
To reset the system, press and hold the reset button for 5 seconds - you'll find it on the main control panel (Part #RST-001).

Q: "What's the operating temperature range?"
The operating range is -10°C to 60°C. Need to know anything specific about temperature monitoring or troubleshooting?

Q: "How do I replace the filter?"
Here's how to replace the filter:\n\n• **First, turn off power** and unplug the unit for safety\n• Remove the front panel by pressing the two side tabs\n• Slide out the old filter (Part #FLT-200) and dispose of it\n• Insert the new filter until you hear it click into place\n• Reattach the panel and power back up\n\nNeed help finding the right replacement filter or have questions about the process?

Your response:"""
                    
                    print(f"\n=== LLM Call ===")
                    print(f"Prompt Length: {len(prompt)} characters")
                    print(f"Context Length: {len(context)} characters")
                    
                    # Generate response
                    response = await asyncio.to_thread(self.llm.invoke, prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    print(f"LLM Response Length: {len(response_text)} characters")
                    print(f"=== End RAG Processing ===\n")
                    
                    # Extract sources from chunks and web links (only if web search was performed)
                    response_web_links = web_links if should_search_web else []
                    sources = self._extract_sources(chunks, response_web_links)
                    
                    return ChatResponse(
                        response=response_text,
                        sources=sources,
                        chunks_found=len(chunks)
                    )
                    
                except Exception as e:
                    print(f"Error generating LLM response: {e}")
                    # Fall through to fallback response
            
            # Step 5: Fallback response when LLM is not available
            fallback_web_links = web_links if should_search_web else []
            response_text = self._generate_fallback_response(context, message, fallback_web_links)
            sources = self._extract_sources(chunks, fallback_web_links)
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                chunks_found=len(chunks)
            )
            
        except Exception as e:
            print(f"Error in chat service: {e}")
            return ChatResponse(
                response="I apologize, but I encountered an error while processing your request. Please try again.",
                sources=[],
                chunks_found=0
            )
    
    def _extract_sources(self, chunks: List[Dict[str, Any]], web_links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract unique source information from chunks and web links"""
        sources = []
        seen_files = set()
        
        # Add document sources
        for chunk in chunks[:5]:  # Limit to top 5 sources
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            if filename != 'Unknown' and filename not in seen_files:
                seen_files.add(filename)
                sources.append({
                    'filename': filename,
                    'similarity_score': round(chunk.get('similarity_score', 0), 3),
                    'source_type': metadata.get('source', 'document'),
                    'upload_time': metadata.get('upload_time', 'unknown')
                })
        
        # Add web link sources
        if web_links:
            for link in web_links[:3]:  # Limit to top 3 web sources
                sources.append({
                    'filename': link.get('title', 'Web Result'),
                    'url': link.get('url', ''),
                    'snippet': link.get('snippet', ''),
                    'source_type': 'web_link',
                    'similarity_score': 0.0  # Web results don't have similarity scores
                })
        
        return sources
    
    def _generate_fallback_response(self, context: str, query: str, web_links: List[Dict[str, Any]] = None) -> str:
        """Generate a concise fallback response when LLM is not available"""
        if context:
            # Extract first relevant chunk for quick response
            lines = context.split('\n')[:5]  # First 5 lines
            summary = ' '.join(lines).strip()[:300]
            response = f"Here's what I found: {summary}..."
            
            # Add web links if available
            if web_links:
                response += "\n\n**Helpful Links:**\n"
                for i, link in enumerate(web_links[:2], 1):
                    response += f"{i}. [{link['title']}]({link['url']})\n"
            
            response += "\n(Note: LLM service temporarily unavailable - showing raw data)"
            return response
        else:
            response = f"I couldn't find specific documentation for '{query}'. "
            
            if web_links:
                response += "However, I found these helpful resources:\n\n"
                for i, link in enumerate(web_links[:3], 1):
                    response += f"{i}. [{link['title']}]({link['url']})\n"
                response += "\nTry these links or upload relevant technical manuals for more specific help."
            else:
                response += "Try uploading relevant technical manuals or rephrasing your question."
            
            return response
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        print("Query cache cleared")
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history (placeholder for future implementation)
        """
        # This would be implemented with a database in a production system
        return []
    
    async def save_conversation(self, conversation_id: str, message: str, response: str) -> bool:
        """
        Save conversation to history (placeholder for future implementation)
        """
        # This would be implemented with a database in a production system
        return True
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get chat service status
        """
        return {
            "service": "ChatService",
            "llm_available": bool(self.llm),
            "openrouter_configured": bool(self.openrouter_api_key),
            "web_search_available": bool(self.web_search_tool),
            "knowledge_base_connected": self.knowledge_base is not None,
            "agent_service_connected": self.agent_service is not None,
            "status": "active"
        }