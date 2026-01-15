from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="./rag_db",
        embedding_function=embeddings
    )

    return vector_db.as_retriever(search_kwargs={"k": 1})