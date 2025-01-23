# RAG-Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot designed to assist users by answering questions based on the context of uploaded documents. It combines state-of-the-art technologies for document processing, embeddings generation, vector search, and natural language generation.

## Features

- **OCR and PDF Parsing**: Extracts text from scanned or text-based PDFs using `PyPDFLoader` and `pytesseract` (OCR).
- **Document Embeddings**: Uses `SentenceTransformer` for generating embeddings of documents and queries.
- **Vector Search**: Implements FAISS for storing and retrieving document embeddings with similarity search.
- **Query Refinement**: An LLM (`llama3.2`) is used to refine user queries for better retrieval performance.
- **Customizable Prompt Templates**: Prompts can be tailored for context-specific assistance.
- **Streamlit Interface**: Allows users to upload PDFs, process them, and update the vector store.
- **FastAPI Backend**: Provides REST APIs for asking questions and refining queries.
- **Sources for Answers**: Responses include document names and page numbers for better traceability.

- ## Technologies Used

### Python Libraries
- **LangChain**: For document loaders, prompts, and retrieval-based question answering.
- **SentenceTransformers**: For generating embeddings.
- **FAISS**: For efficient similarity search.
- **pytesseract**: For OCR text extraction.
- **FastAPI**: For creating the API server.
- **Streamlit**: For creating the Admin interface.

### Frontend
- **Next.js**: For building the frontend of the web application, providing a dynamic and interactive user interface.

### Model:
- **dunzhang/stella_en_1.5B_v5**: for embeddings.
- **llama-3.2**: for query refinement and responses.

## Clone the Repository

1. To clone the repository, run the following commands in your terminal:

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot


