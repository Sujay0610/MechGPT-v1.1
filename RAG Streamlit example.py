import streamlit as st
import os
import tempfile
import gc
import base64
import time

from src.agentic_rag.tools.custom_tool import DocumentSearchTool as SimpleDocumentSearchTool
from src.agentic_rag.tools.custom_tool_complex import DocumentSearchTool as ComplexDocumentSearchTool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource




def load_llm():
    """Load the LLM for direct use without CrewAI wrapper"""
    return ChatOpenAI(
        model="google/gemma-3-27b-it:free",
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1/",
    )

# ===========================
#   Direct RAG Implementation
# ===========================
def direct_rag_query(query, pdf_tool):
    """Direct RAG implementation without complex agent workflow"""
    llm = load_llm()
    
    # Initialize web search
    search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPER_API_KEY"))
    
    print(f"\n=== RAG Query Processing ===")
    print(f"User Query: {query}")
    
    # Try PDF search first
    pdf_results = ""
    if pdf_tool:
        try:
            print("Searching PDF documents...")
            pdf_results = pdf_tool.search(query)
            print(f"PDF Results Length: {len(pdf_results)} characters")
        except Exception as e:
            print(f"PDF search error: {str(e)}")
            st.error(f"PDF search error: {str(e)}")
            pdf_results = ""
    
    # If no good PDF results, try web search
    web_results = ""
    if not pdf_results or len(pdf_results.strip()) < 50:
        try:
            print("Performing web search...")
            web_results = search.run(query)
            print(f"Web Results Length: {len(web_results)} characters")
        except Exception as e:
            print(f"Web search error: {str(e)}")
            st.error(f"Web search error: {str(e)}")
            web_results = ""
    
    # Combine results
    context = ""
    if pdf_results:
        context += f"Document Information:\n{pdf_results}\n\n"
    if web_results:
        context += f"Web Search Results:\n{web_results}\n\n"
    
    if not context:
        print("No context found for query")
        return "I'm sorry, I couldn't find any relevant information for your query."
    
    # Create prompt for direct LLM response tailored for technicians/maintenance personnel
    prompt = f"""You are an expert technical assistant helping maintenance technicians and field service personnel. Based on the following technical documentation, provide practical, actionable information to help with equipment maintenance, troubleshooting, and operations.

Technical Documentation and Information:
{context}

Technician Query: {query}

Instructions for your response:
- Focus on practical, hands-on information 
- Include specific procedures, part numbers, safety warnings, and step-by-step instructions when available
- Use clear, technical language 
- If the information is insufficient for safe operation, clearly state what additional documentation or expertise is needed
- You have to make the response concise and to the point like a chatbot answers [Important]
- If the question is not related to the provided technical documentation or information, politely respond that you are programmed to only answer questions related to the provided context.

Technical Response:"""
    
    print(f"\n=== LLM Call ===")
    print(f"Prompt Length: {len(prompt)} characters")
    print(f"Context Length: {len(context)} characters")
    
    try:
        print("Calling LLM...")
        response = llm.invoke(prompt)
        print(f"LLM Response Length: {len(response.content)} characters")
        print(f"=== End RAG Processing ===\n")
        return response.content
    except Exception as e:
        print(f"LLM error: {str(e)}")
        st.error(f"LLM error: {str(e)}")
        return "I encountered an error while processing your request. Please try again."

# ===========================
#   Streamlit Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def format_chat_history(messages, max_messages=5):
    """Format recent chat history for context."""
    if not messages:
        return "No previous conversation."
    
    # Get the last few messages for context (excluding the current one)
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted_history = "Recent conversation:\n"
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        formatted_history += f"{role}: {content}\n"
    
    return formatted_history

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("📋 Technical Documentation")
    
    # Show conversation statistics
    if st.session_state.messages:
        num_exchanges = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        st.metric("Conversation Exchanges", num_exchanges)
    
    # Document source selection
    doc_source = st.radio(
        "Choose technical documentation source:",
        ["Upload New Manual/Document", "Use Existing Technical Library"],
        help="Upload new technical documentation or select from previously indexed manuals and guides"
    )
    
    if doc_source == "Upload New Manual/Document":
        uploaded_file = st.file_uploader("Upload Technical Documentation (PDF)", type=["pdf"])

        tool_choice = st.radio(
            "Choose Document Analysis Type:",
            ('Simple PDF Processing', 'Complex Document Analysis (with images)'),
            key='tool_choice_radio',
            on_change=reset_chat # Reset chat and tool when choice changes
        )
        st.session_state.tool_choice = 'simple' if tool_choice == 'Simple PDF Processing' else 'complex'
        
        # Show which RAG system will be used
        if tool_choice == 'Simple PDF Processing':
            st.info("🔧 **RAG System:** Chroma (Local, Fast)")
        else:
            st.info("🔧 **RAG System:** GroundX (Cloud, Advanced)")
            st.caption("⏱️ Note: GroundX processing takes longer but handles images and complex layouts better.")

        st.markdown("""
            <style>
            .stRadio > label > div:first-child {
                padding-right: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            # If there's a new file and we haven't set pdf_tool yet...
                    # If there's a new file and we haven't set pdf_tool yet...
            # If there's a new file and we haven't set pdf_tool yet, or if the tool choice has changed
            if st.session_state.pdf_tool is None or st.session_state.get('previous_tool_choice') != tool_choice:
                st.session_state.previous_tool_choice = tool_choice # Store current choice for next check
                # Clear existing tool to force re-initialization
                st.session_state.pdf_tool = None

                # tool_choice is already defined above

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    try:
                        if tool_choice == 'Simple PDF Processing':
                            with st.spinner("Indexing PDF with Chroma... Please wait..."):
                                st.session_state.pdf_tool = SimpleDocumentSearchTool(file_path=temp_file_path)
                            st.success("✅ PDF indexed with Chroma! Ready to chat.")
                        else:
                            with st.spinner("Uploading and processing document with GroundX... This may take a few minutes..."):
                                st.session_state.pdf_tool = ComplexDocumentSearchTool(file_path=temp_file_path)
                            st.success("✅ PDF fully processed with GroundX! Ready for complex document analysis.")
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.session_state.pdf_tool = None

            # Optionally display the PDF in the sidebar
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    else:  # Use Existing GroundX Document
        st.subheader("📚 Bucket Management")
        
        if st.button("🔄 Refresh Bucket List"):
            if 'existing_buckets' in st.session_state:
                del st.session_state.existing_buckets
        
        # Load existing buckets if not already loaded
        if 'existing_buckets' not in st.session_state:
            with st.spinner("Loading existing buckets..."):
                try:
                     st.session_state.existing_buckets = ComplexDocumentSearchTool.list_existing_buckets()
                except Exception as e:
                    st.error(f"❌ Error loading existing buckets: {str(e)}")
                    st.session_state.existing_buckets = []
        
        # Display existing buckets
        if st.session_state.get('existing_buckets'):
            bucket_options = {}
            
            for bucket in st.session_state.existing_buckets:
                doc_count = len([doc for doc in bucket.get('documents', []) if doc['status'] == 'complete'])
                display_name = f"{bucket['bucket_name']} ({doc_count} documents)"
                bucket_options[display_name] = {
                    'bucket_id': bucket['bucket_id'],
                    'bucket_name': bucket['bucket_name'],
                    'documents': bucket.get('documents', []),
                    'created_at': bucket['created_at']
                }
            
            if bucket_options:
                selected_bucket = st.selectbox(
                    "Select a bucket:",
                    list(bucket_options.keys()),
                    help="Choose a bucket to work with"
                )
                
                if selected_bucket:
                    bucket_info = bucket_options[selected_bucket]
                    
                    # Show bucket details
                    st.write(f"**Bucket:** {bucket_info['bucket_name']}")
                    st.write(f"**Documents:** {len([doc for doc in bucket_info['documents'] if doc['status'] == 'complete'])} completed")
                    
                    # Show documents in bucket
                    if bucket_info['documents']:
                        st.write("**Documents in this bucket:**")
                        for doc in bucket_info['documents']:
                            status_emoji = "✅" if doc['status'] == 'complete' else "⏳" if doc['status'] == 'processing' else "❌"
                            st.write(f"{status_emoji} {doc['name']} ({doc['status']})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📖 Use This Bucket", use_container_width=True):
                            try:
                                bucket_id = bucket_info['bucket_id']
                                st.session_state.pdf_tool = ComplexDocumentSearchTool(bucket_id=bucket_id)
                                st.session_state.tool_type = "complex"
                                st.session_state.pdf_processed = True
                                st.session_state.current_bucket_id = bucket_id
                                
                                st.success(f"✅ Using bucket: {bucket_info['bucket_name']}")
                                st.info("🚀 **RAG System**: (Existing Bucket)")
                                
                            except Exception as e:
                                st.error(f"❌ Error loading bucket: {str(e)}")
                    
                    with col2:
                        # Add document to existing bucket
                        uploaded_file_to_bucket = st.file_uploader(
                            "Add PDF to this bucket:", 
                            type=["pdf"], 
                            key=f"upload_to_{bucket_info['bucket_id']}"
                        )
                        
                        if uploaded_file_to_bucket and st.button("➕ Add to Bucket", use_container_width=True):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_file_path = os.path.join(temp_dir, uploaded_file_to_bucket.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_file_to_bucket.getvalue())
                                
                                try:
                                    with st.spinner(f"Adding {uploaded_file_to_bucket.name} to bucket..."):
                                        process_id = ComplexDocumentSearchTool.add_document_to_bucket(
                                            bucket_info['bucket_id'], 
                                            temp_file_path
                                        )
                                        ComplexDocumentSearchTool.wait_for_document_processing(process_id)
                                    
                                    st.success(f"✅ Added {uploaded_file_to_bucket.name} to bucket!")
                                    # Refresh bucket list
                                    if 'existing_buckets' in st.session_state:
                                        del st.session_state.existing_buckets
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error adding document: {str(e)}")
            else:
                st.info("📝 No buckets found.")
        else:
            st.info("📝 No existing buckets found.")
        
        # Option to create new bucket
        st.markdown("---")
        st.subheader("🆕 Create New Bucket")
        
        new_bucket_name = st.text_input("Bucket name:", placeholder="Enter a name for your new bucket")
        uploaded_file_new_bucket = st.file_uploader("Upload first PDF:", type=["pdf"], key="new_bucket_upload")
        
        if new_bucket_name and uploaded_file_new_bucket and st.button("🚀 Create Bucket"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file_new_bucket.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file_new_bucket.getvalue())
                
                try:
                    with st.spinner(f"Creating bucket '{new_bucket_name}' and processing document..."):
                        # Create bucket with custom name
                        bucket_id = ComplexDocumentSearchTool.create_bucket_with_name(new_bucket_name)
                        
                        # Add document to the new bucket
                        process_id = ComplexDocumentSearchTool.add_document_to_bucket(bucket_id, temp_file_path)
                        ComplexDocumentSearchTool.wait_for_document_processing(process_id)
                        
                        # Create tool with the new bucket
                        st.session_state.pdf_tool = ComplexDocumentSearchTool(bucket_id=bucket_id)
                        st.session_state.tool_type = "complex"
                        st.session_state.pdf_processed = True
                        st.session_state.current_bucket_id = bucket_id
                    
                    st.success(f"✅ Created bucket '{new_bucket_name}' and processed {uploaded_file_new_bucket.name}!")
                    st.info("🚀 **RAG System**: GroundX (New Bucket)")
                    
                    # Refresh bucket list
                    if 'existing_buckets' in st.session_state:
                        del st.session_state.existing_buckets
                    
                except Exception as e:
                    st.error(f"❌ Error creating bucket: {str(e)}")
    
    # Show conversation stats
    if st.session_state.messages:
        st.markdown("---")
        st.markdown(f"💬 **Conversation:** {len(st.session_state.messages)//2} exchanges")


    st.markdown("---")
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        reset_chat()
        st.rerun()
    
    if st.session_state.messages:
        st.caption("💡 Tip: I remember our conversation context for better responses!")

# ===========================
#   Main Chat Interface
# ===========================
st.markdown("""
    # 🔧 Technical Maintenance Assistant
    ### AI-powered technical support for maintenance technicians and field service personnel
""")
st.markdown("---")

# Show welcome message if no messages yet
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.pdf_tool:
            st.markdown("🔧 Hello! I'm your technical maintenance assistant. I'm ready to help you with equipment manuals, troubleshooting procedures, maintenance schedules, and operational guidance. What technical information do you need?")
        else:
            st.markdown("🔧 Hello! I'm your technical maintenance assistant. Please upload technical documentation, equipment manuals, or maintenance guides in the sidebar to get started. I'll help you find procedures, troubleshooting steps, and operational information!")

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.pdf_tool:
    prompt = st.chat_input("🔧 Ask about procedures, troubleshooting, maintenance, or safety information...")
else:
    prompt = st.chat_input("Please upload technical documentation first to start...")
    if prompt:
        st.warning("⚠️ Please upload technical documentation in the sidebar before asking questions.")
        prompt = None  # Prevent processing

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response using direct RAG (no chat history needed as Streamlit handles UI context)
            response = direct_rag_query(prompt, st.session_state.pdf_tool)
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
