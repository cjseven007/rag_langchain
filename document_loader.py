from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Function to load documents into a vector store


def load_vector_store(doc_path):
    doc_loader = PyPDFLoader(doc_path)
    documents = doc_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db
