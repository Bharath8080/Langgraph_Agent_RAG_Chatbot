"""
ui.py - Streamlit UI for the LangGraph RAG Agent with Gemini
"""
import streamlit as st
import os
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Import core functionality from scripts
from scripts.main import (
    create_agent,
    setup_environment,
    AgentState,
    BaseMessage,
    HumanMessage,
    AIMessage
)
from scripts.rag import load_documents_from_files, create_vector_index

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'config' not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "streamlit-thread"}}

def show_api_config(api_keys: Dict[str, str]) -> bool:
    """Show API configuration and return status"""
    st.sidebar.header("🔑 API Configuration")
    
    # Show API key status
    with st.sidebar.expander("API Key Status"):
        google_status = "✅" if api_keys["google_api_key"] else "❌"
        tavily_status = "✅" if api_keys["tavily_api_key"] else "❌"
        
        st.write(f"{google_status} Google API Key")
        st.write(f"{tavily_status} Tavily API Key")
        
        if not all(api_keys.values()):
            st.error("Please provide all required API keys in the .env file")
            st.info("""
            Create a .env file in the same directory with these variables:
            
            GOOGLE_API_KEY=your_key_here
            TAVILY_API_KEY=your_key_here
            """)
            return False
    
    # Add disclaimer expander
    with st.sidebar.expander("⚠️ Important Notes"):
        st.markdown("""
        1. **Processing Time**: Larger documents may take longer to process. 
           - The system needs time to analyze and index the content.
           - Some context might be missed with very large documents.
        
        2. **PDF Compatibility**: 
           - Some PDFs with complex layouts, scanned pages, or images may not be processed accurately.
           - For best results, use text-based PDFs with clear formatting.
        """)
    
    return True

def show_document_upload():
    """Show document upload interface"""
    st.sidebar.header("📄 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, TXT, or MD files to create your knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files and st.sidebar.button("🔄 Process Documents"):
        with st.spinner("Processing documents..."):
            documents = load_documents_from_files(uploaded_files)
            if documents:
                retriever, vectordb = create_vector_index(documents)
                if retriever and vectordb:
                    st.session_state.retriever = retriever
                    st.session_state.vectordb = vectordb
                    st.sidebar.success(f"✅ Processed {len(documents)} documents")
                else:
                    st.sidebar.error("❌ Failed to create vector index")

def show_agent_controls():
    """Show agent initialization controls"""
    st.sidebar.header("🤖 Agent Controls")
    
    # Initialize agent button
    if st.sidebar.button("🚀 Initialize Agent"):
        if not st.session_state.get('retriever'):
            st.sidebar.warning("⚠ Please upload and process documents first!")
        else:
            with st.spinner("Initializing agent..."):
                st.session_state.agent = create_agent(st.session_state.retriever)
                st.sidebar.success("✅ Agent initialized successfully!")
    
    # Clear chat button
    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def show_chat_interface():
    """Show the main chat interface"""
    # st.header("💬 Chat with your Agent")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not st.session_state.agent:
            st.error("❌ Please initialize the agent first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=st.session_state.config
                    )
                    
                    # Get the last AI message
                    last_message = next((m for m in reversed(result["messages"])
                                       if isinstance(m, AIMessage)), None)
                    
                    if last_message:
                        response = last_message.content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("No response generated")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LangGraph RAG Agent with Gemini",
        page_icon="🤖",
        layout="wide"
    )
    
    # st.title("🤖 LangGraph RAG Agent with Gemini")
    st.markdown(f"<h2 style='color: #07f576;'>💬LangGraph RAG Agent with Gemini</h3>", unsafe_allow_html=True)
    # st.markdown("An intelligent agent that can search your documents and the web using Google's Gemini models")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup environment and get API keys
    api_keys_ready, api_keys = setup_environment()
    
    # Show API configuration and check if all keys are present
    if not show_api_config(api_keys):
        st.warning("⚠ Please provide all required API keys in the .env file")
        return
    
    # Show document upload section
    show_document_upload()
    
    # Show agent controls
    show_agent_controls()
    
    # Show chat interface
    show_chat_interface()

if __name__ == "__main__":
    main()
