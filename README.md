# Langgraph Agentic RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot built with LangGraph, designed to provide accurate and context-aware responses by leveraging external knowledge sources.

<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
    <div style="flex: 1; min-width: 300px;">
        <img src="https://github.com/Bharath8080/Langgraph_Agentic_RAG_ChatBOT/raw/main/assets/agent_workflow.png" alt="Agent Workflow" style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    <div style="flex: 1; min-width: 300px;">
        <img src="https://github.com/Bharath8080/Langgraph_Agentic_RAG_ChatBOT/raw/main/assets/rag%20agent%20flow%20minimal.png" alt="RAG Agent Flow Minimal" style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
</div>

## 🌐 Live Demo

Try out the live demo of the application: [LangGraph Agentic RAG Chatbot](https://langgraphagenticragchatbot.streamlit.app/)

## 🚀 Features

- **Agentic Architecture**: Built with LangGraph for complex, stateful conversation flows
- **RAG Integration**: Retrieval-Augmented Generation for accurate, up-to-date responses
- **Multi-LLM Support**: Compatible with various language models
- **Streamlit UI**: User-friendly web interface for easy interaction
- **Tracing & Monitoring**: Integrated with LangSmith for observability

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bharath8080/Langgraph_Agentic_RAG_ChatBOT.git
   cd Langgraph_Agentic_RAG_ChatBOT
   ```

2. **Set up virtual environment (recommended using `uv` for speed)**
   ```bash
   pip install uv
   uv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory with the following content:
   ```env
   # API Keys
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   COHERE_API_KEY=your_cohere_api_key

   # LangSmith Configuration
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=langgraph-rag-agent
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

   > **Note**: Replace the placeholder values with your actual API keys. Do not commit the `.env` file to version control.

## 🚀 Usage

1. **Start the Streamlit application**
   ```bash
   streamlit run ui.py
   ```

2. **Open your browser** and navigate to the URL provided in the terminal (typically `http://localhost:8501`)

## 📊 Features in Action

- **Natural Language Queries**: Ask questions in natural language
- **Context-Aware Responses**: The bot maintains conversation context
- **Source Attribution**: View sources for the information provided
- **Conversation History**: Review past interactions

## 🤖 Technologies Used

- **LangGraph**: For building agentic workflows
- **LangChain**: For orchestration of LLM components
- **Streamlit**: For the web interface
- **Tavily**: For web search capabilities
- **Cohere**: For embeddings and language understanding
- **LangSmith**: For monitoring and tracing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the LangChain team for their amazing tools
- The open-source community for their contributions
- All the model providers that make this project possible
