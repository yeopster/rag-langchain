# Intelligent Trading Q&A System

A web-based application that allows users to ask questions about trading and receive contextually relevant answers. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and LangChain to integrate retrieval with a large language model (LLM) for answer generation. A Streamlit interface provides a user-friendly front end.

This project demonstrates expertise in natural language processing (NLP), vector databases, and LLM-powered applications, making it ideal for portfolio showcasing.

## Features
- Extracts text from PDF technical manuals for processing.
- Uses RAG to combine document retrieval with LLM-based answer generation.
- Built with LangChain for orchestrating the RAG pipeline.
- Stores document embeddings in a FAISS vector database for fast similarity search.
- Interactive Streamlit web interface for querying manuals.
- Custom prompt engineering to ensure clean, accurate answers.

## Demo
https://rag-langchain-av7xqfe4nbqpnrx3btbtpu.streamlit.app/

## Project Structure
'''bash
rag-langchain/ ├── data/                    # Store PDFs and processed data │   ├── processed/           # Extracted text files │   └── vector_store/        # FAISS vector store ├── scripts/                 # Processing scripts │   ├── extract_text.py      # Extract text from PDFs │   ├── chunk_documents.py   # Split text into chunks │   ├── create_vector_store.py # Create FAISS vector store │   └── rag_pipeline.py      # RAG pipeline with LangChain ├── app.py                   # Streamlit web interface ├── requirements.txt         # Python dependencies ├── .gitignore               # Git ignore file └── README.md                # Project documentation
