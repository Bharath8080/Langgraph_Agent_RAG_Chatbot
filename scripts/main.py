"""
main.py - Main functionality for the LangGraph RAG Agent with Gemini
"""
from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from .rag import load_documents_from_files, create_vector_index
from .tools import Tools

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

def create_agent(retriever=None):
    """Create the LangGraph agent"""
    # Initialize tools
    tools = Tools(retriever=retriever)
    web_search_tool = tools.web_search_tool
    rag_search_tool = tools.rag_search_tool
    
    # Initialize Gemini LLMs
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    ).with_structured_output(RouteDecision)
    
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    ).with_structured_output(RagJudge)
    
    answer_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
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
    return agent

def setup_environment() -> tuple[bool, dict[str, str]]:
    """Setup API keys and environment variables"""
    # Try to get API keys from Streamlit secrets first
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
            tavily_api_key = st.secrets.get("TAVILY_API_KEY", "")
    except:
        google_api_key = ""
        tavily_api_key = ""
    
    # Fall back to environment variables if not in Streamlit
    google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    
    # Check if all required API keys are present
    all_keys_present = bool(google_api_key and tavily_api_key)
    
    # Set environment variables (in case they weren't already set)
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    return all_keys_present, {
        "google_api_key": google_api_key,
        "tavily_api_key": tavily_api_key
    }
