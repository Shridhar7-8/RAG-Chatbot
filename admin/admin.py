from langchain.schema import Document
from pdf2image import convert_from_path
import pytesseract
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class MyEmbeddings:
    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")

    def __call__(self, text):
        if isinstance(text, str):
            return self.model.encode(text)

    def embed_documents(self, documents):
        return self.model.encode(documents)

    def embed_query(self, query):
        return self.model.encode(query)


embedding_model = MyEmbeddings()

VECTOR_STORE_FILE = "vector_store.bin"

def requires_ocr(pdf_path):
    """Check if OCR is needed by attempting to extract text."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages[:5]:
            if page.page_content.strip():
                return False
        return True
    except Exception as e:
        st.write(f"Error checking for OCR: {e}")
        return True

def perform_ocr(pdf_path):
    """Perform OCR on a PDF and return extracted text as Document objects."""
    images = convert_from_path(pdf_path)  
    docs = []
    for i, image in enumerate(images):

        text = pytesseract.image_to_string(image)
        docs.append(Document(page_content=text, metadata={"page": i + 1,"source":os.path.basename(pdf_path)}))
    print(docs)
    return docs

def load_pdf(file_path):
    """Load and split text-based PDF into pages."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    #base_url = "https://nmicps.in/home/bannerimages"
    pdf_name = os.path.basename(file_path)
    #full_pdf_url = f"{base_url}/{pdf_name}"
    for page in pages:
        page.metadata["source"] = pdf_name
    print(pages)
    return pages

def split_text(pages, chunk_size, chunk_overlap):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def load_or_create_vector_store():
    """Load or create the vector store."""
    if os.path.exists(VECTOR_STORE_FILE):
        st.write("Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_FILE, embedding_model, allow_dangerous_deserialization=True)
    else:
        st.write("No existing vector store found. A new one will be created.")
        return None

def update_vector_store(vector_store, new_docs):
    """Update the vector store with new documents."""
    if vector_store is None:
        st.write("Creating a new vector store...")
        vector_store = FAISS.from_documents(new_docs, embedding_model)
    else:
        st.write("Adding documents to the existing vector store...")
        vector_store.add_documents(new_docs)

    st.write("Saving updated vector store...")
    vector_store.save_local(VECTOR_STORE_FILE)
    return vector_store

def process_uploaded_file(file_path):
    """Process uploaded file with or without OCR."""
    if requires_ocr(file_path):
        st.write("Performing OCR on the document...")
        return perform_ocr(file_path)
    else:
        st.write("Loading text from the document...")
        return load_pdf(file_path)

def main():
    st.title("Upload PDFs to Existing Vector Store")

 
    vector_store = load_or_create_vector_store()

    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getvalue())

            st.write(f"Processing file: {file_name}")
            docs = process_uploaded_file(file_name)
            st.write(f"Number of pages in {file_name}: {len(docs)}")
            docs = split_text(docs, 1000, 200)
            st.write(f"Number of documents extracted from {file_name}: {len(docs)}")
            all_docs.extend(docs)

        st.write(f"Total new documents: {len(all_docs)}")

        if all_docs:
            st.write("Updating the vector store with new documents...")
            vector_store = update_vector_store(vector_store, all_docs)
            st.write("Vector store updated successfully.")
        else:
            st.write("No documents to process. Please upload valid PDF files.")

if __name__ == "__main__":
    main()
