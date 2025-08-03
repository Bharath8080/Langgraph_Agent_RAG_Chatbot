import streamlit as st
import os
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field
import tempfile
import shutil

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import CohereEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Pydantic schemas for structured output
class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool

# State definition
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "answer", "end"]
    rag: str
    web: str

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'config' not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit-thread"}}

def load_environment_vars():
    """Load environment variables from .env file"""
    from dotenv import load_dotenv
    import os
    
    # Load .env file from the same directory as the script
    env_path = os.path.join(os.path.dirname(_file_), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded environment variables from: {env_path}")
        return True
    else:
        print(f"⚠ Warning: .env file not found at {env_path}")
        return False

def setup_environment():
    """Setup API keys and environment variables"""
    st.sidebar.header("🔑 API Configuration")
    
    # Load environment variables from .env file
    env_loaded = load_environment_vars()
    
    # Get API keys from environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    cohere_api_key = os.getenv("COHERE_API_KEY", "")
    
    # Check if all required API keys are present without showing status
    all_keys_present = bool(google_api_key and tavily_api_key and cohere_api_key)
    
    # Only show warnings if something is wrong
    if not all_keys_present:
        with st.sidebar:
            st.warning("⚠ Some API keys are missing")
            with st.expander("Show missing keys"):
                if not google_api_key:
                    st.error("Google API Key is missing")
                if not tavily_api_key:
                    st.error("Tavily API Key is missing")
                if not cohere_api_key:
                    st.error("Cohere API Key is missing")
                
                st.info("""
                Create a .env file in the same directory with these variables:
                
                GOOGLE_API_KEY=your_key_here
                TAVILY_API_KEY=your_key_here
                COHERE_API_KEY=your_key_here
                
                """)
    
    # Set environment variables (in case they weren't already set)
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
    if cohere_api_key:
        os.environ["COHERE_API_KEY"] = cohere_api_key
    
    # Check if all required API keys are present
    all_keys_present = bool(google_api_key and tavily_api_key and cohere_api_key)
    
    return all_keys_present

def load_documents_from_files(uploaded_files) -> List[Document]:
    """Load documents from uploaded files"""
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temp directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load document based on file type
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif uploaded_file.name.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif uploaded_file.name.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                documents.extend(loader.load())
                pass  # Don't show success message for each file
                
            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    return documents

def create_vector_index(documents: List[Document]):
    """Create a vector index from documents using Cohere embeddings with FAISS"""
    
    if not documents:
        st.warning("No documents to process!")
        return None
        
    with st.spinner("Processing documents..."):
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    chunks = text_splitter.split_documents(documents)
    
    try:
        # Create embeddings using Cohere
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            user_agent="langchain-app"
        )
        
        # Create FAISS vector store
        with st.spinner("Creating FAISS index..."):
            vectordb = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            # Create retriever
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            st.session_state.retriever = retriever
            st.session_state.vectordb = vectordb  # Store the vector store in session state
            
            pass  # Don't show success message
            return retriever
            
    except Exception as e:
        st.error(f"Error creating vector index: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_tools():
    """Create RAG and web search tools"""
    
    @tool
    def web_search_tool(query: str) -> str:
        """Up-to-date web info via Tavily"""
        try:
            tavily = TavilySearch(max_results=3, topic="general")
            result = tavily.invoke({"query": query})
            
            if isinstance(result, dict) and 'results' in result:
                formatted_results = []
                for item in result['results']:
                    title = item.get('title', 'No title')
                    content = item.get('content', 'No content')
                    url = item.get('url', '')
                    formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
                
                return "\n\n".join(formatted_results) if formatted_results else "No results found"
            else:
                return str(result)
        except Exception as e:
            return f"WEB_ERROR::{e}"
    
    @tool
    def rag_search_tool(query: str) -> str:
        """Top-3 chunks from KB (empty string if none)"""
        try:
            if not st.session_state.retriever:
                return "No knowledge base available"
            
            docs = st.session_state.retriever.invoke(query)
            return "\n\n".join(d.page_content for d in docs) if docs else ""
        except Exception as e:
            return f"RAG_ERROR::{e}"
    
    return web_search_tool, rag_search_tool

def create_agent():
    """Create the LangGraph agent"""
    
    web_search_tool, rag_search_tool = create_tools()
    
    # Initialize Gemini LLMs
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    ).with_structured_output(RouteDecision)
    
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    ).with_structured_output(RagJudge)
    
    answer_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7
    )
    
    # Node functions
    def router_node(state: AgentState) -> AgentState:
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        
        messages = [
            ("system", (
                "You are a router that decides how to handle user queries:\n"
                "- Use 'end' for pure greetings/small-talk (also provide a 'reply')\n"
                "- Use 'rag' when knowledge base lookup is needed\n"
                "- Use 'answer' when you can answer directly without external info"
            )),
            ("user", query)
        ]
        
        result: RouteDecision = router_llm.invoke(messages)
        
        out = {"messages": state["messages"], "route": result.route}
        if result.route == "end":
            out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
        return out
    
    def rag_node(state: AgentState) -> AgentState:
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        
        chunks = rag_search_tool.invoke({"query": query})
        
        judge_messages = [
            ("system", (
                "You are a judge evaluating if the retrieved information is sufficient "
                "to answer the user's question. Consider both relevance and completeness."
            )),
            ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
        ]
        
        verdict: RagJudge = judge_llm.invoke(judge_messages)
        
        return {
            **state,
            "rag": chunks,
            "route": "answer" if verdict.sufficient else "web"
        }
    
    def web_node(state: AgentState) -> AgentState:
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        snippets = web_search_tool.invoke({"query": query})
        return {**state, "web": snippets, "route": "answer"}
    
    def answer_node(state: AgentState) -> AgentState:
        user_q = next((m.content for m in reversed(state["messages"])
                       if isinstance(m, HumanMessage)), "")
        
        ctx_parts = []
        if state.get("rag"):
            ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
        if state.get("web"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
        
        context = "\n\n".join(ctx_parts) if ctx_parts else "No external context available."
        
        prompt = f"""Please answer the user's question using the provided context.

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""
        
        ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=ans)]
        }
    
    # Routing functions
    def from_router(st: AgentState) -> Literal["rag", "answer", "end"]:
        return st["route"]
    
    def after_rag(st: AgentState) -> Literal["answer", "web"]:
        return st["route"]
    
    def after_web(_) -> Literal["answer"]:
        return "answer"
    
    # Build graph
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)
    
    g.set_entry_point("router")
    g.add_conditional_edges("router", from_router,
                            {"rag": "rag_lookup", "answer": "answer", "end": END})
    g.add_conditional_edges("rag_lookup", after_rag,
                            {"answer": "answer", "web": "web_search"})
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)
    
    agent = g.compile(checkpointer=MemorySaver())
    st.session_state.agent = agent
    
    return agent

def main():
    st.set_page_config(
        page_title="🤖 LangGraph RAG Agent with Gemini",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LangGraph RAG Agent with Gemini")
    st.markdown("An intelligent agent that can search your documents and the web using Google's Gemini models")
    
    # Setup environment
    api_keys_ready = setup_environment()
    
    if not api_keys_ready:
        st.warning("⚠ Please provide both Google API Key and Tavily API Key in the sidebar to continue.")
        st.stop()
    
    # Sidebar for document upload
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
                create_vector_index(documents)
                st.sidebar.success(f"✅ Processed {len(documents)} documents")
    
    # Create agent button
    if st.sidebar.button("🚀 Initialize Agent"):
        if not st.session_state.retriever:
            st.sidebar.warning("⚠ Please upload and process documents first!")
        else:
            with st.spinner("Initializing agent..."):
                create_agent()
                st.sidebar.success("✅ Agent initialized successfully!")
    
    # Chat interface
    st.header("💬 Chat with your Agent")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not st.session_state.agent:
            st.error("❌ Please initialize the agent first!")
            st.stop()
        
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
    
    # Clear chat button
    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    

if __name__ == "__main__":
    main()
