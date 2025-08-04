"""
tools.py - Tool definitions for the RAG agent
"""
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

class Tools:
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.web_search_tool = self._create_web_search_tool()
        self.rag_search_tool = self._create_rag_search_tool()
    
    def _create_web_search_tool(self):
        """Create web search tool using Tavily"""
        @tool
        def web_search_tool(query: str) -> str:
            """Up-to-date web info via Tavily"""
            try:
                tavily = TavilySearch(max_results=3, search_depth="advanced",topic="general")
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
        
        return web_search_tool
    
    def _create_rag_search_tool(self):
        """Create RAG search tool"""
        @tool
        def rag_search_tool(query: str) -> str:
            """Top-3 chunks from KB (empty string if none)"""
            try:
                if not self.retriever:
                    return "No knowledge base available"
                
                docs = self.retriever.invoke(query)
                return "\n\n".join(d.page_content for d in docs) if docs else ""
            except Exception as e:
                return f"RAG_ERROR::{e}"
        
        return rag_search_tool
    
    def get_tools(self):
        """Return both tools"""
        return [self.web_search_tool, self.rag_search_tool]
