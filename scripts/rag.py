"""
rag.py - RAG (Retrieval Augmented Generation) related functionality
"""
from typing import List, Optional, Any, Tuple
from pathlib import Path
import tempfile
import os
import logging
import fitz  # PyMuPDF

# Suppress PyMuPDF warnings
logging.getLogger('fitz').setLevel(logging.ERROR)
logging.getLogger('pymupdf').setLevel(logging.ERROR)

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
                    print(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                documents.extend(loader.load())
                
            except Exception as e:
                print(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    return documents

def create_vector_index(documents: List[Document]) -> Tuple[Any, Any]:
    """Create a vector index from documents using HuggingFace embeddings with FAISS"""
    
    if not documents:
        print("No documents to process!")
        return None, None
        
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    try:
        # Create embeddings using HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vector store
        vectordb = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        return retriever, vectordb
            
    except Exception as e:
        print(f"Error creating vector index: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None
