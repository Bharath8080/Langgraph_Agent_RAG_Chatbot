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
    """Load documents from uploaded files with enhanced metadata"""
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
                
                # Load and enhance documents with metadata
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'source': uploaded_file.name,
                        'file_type': os.path.splitext(uploaded_file.name)[1],
                        'total_pages': len(loaded_docs) if hasattr(loader, 'load_and_split') else 1
                    })
                documents.extend(loaded_docs)
                
            except Exception as e:
                print(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    return documents

def create_vector_index(documents: List[Document]) -> Tuple[Any, Any]:
    """Create a vector index from documents using enhanced settings for better accuracy"""
    
    if not documents:
        print("No documents to process!")
        return None, None
        
    # Enhanced text splitting with better chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Smaller chunks for more focused retrieval
        chunk_overlap=128,  # Increased overlap for better context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Better separation of content
        is_separator_regex=False
    )
    
    # Process and split documents
    chunks = text_splitter.split_documents(documents)
    
    # Add position information to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 1
    
    try:
        # Use a more powerful embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Better quality embeddings
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        
        # Create FAISS vector store with better indexing
        vectordb = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
            distance_strategy="COSINE"  # Better for semantic search
        )
        
        # Enhanced retriever with MMR (Maximal Marginal Relevance) for better diversity
        retriever = vectordb.as_retriever(
            search_type="mmr",  # Use MMR for better result diversity
            search_kwargs={
                'k': 5,  # Retrieve more documents
                'fetch_k': 20,  # Larger candidate pool for MMR
                'lambda_mult': 0.5  # Balance between relevance and diversity
            }
        )
        return retriever, vectordb
            
    except Exception as e:
        print(f"Error creating vector index: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None
