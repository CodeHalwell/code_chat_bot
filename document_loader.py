import os
import dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
api_key = os.getenv("OPENAI")

class DocumentLoader:
    def __init__(self, file_path: str = None, url: str = None):
        self.file_path = file_path
        self.url = url

    def load_csv(self, file_path: str) -> list:
        """ Load a CSV file and return its contents. """
        loader = CSVLoader(file_path)
        return loader.load()

    def load_pdf(self, file_path: str) -> str:
        """ Load a PDF file and return its text. """
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_text(self, file_path: str) -> str:
        """ Load a text file and return its contents. """
        loader = TextLoader(file_path)
        return loader.load()

    def load_web(self, url: str) -> str:
        """ Load a webpage and return its text content. """
        loader = WebBaseLoader(url)
        return loader.load()

class VectorDatabase:
    def __init__(self, collection_name: str = None, distance_metric: str = "L2"):
        self.collection_name = collection_name
        self.distance_metric = distance_metric

    def split_document(self, document: str, chunk_size: int = 500, chunk_overlap: int = 75) -> list:
        """ Split the document into smaller parts. """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_document(document)

    def embed_upload(self, collection_name: str, split_documents: list) -> Chroma:
        """ Upload document segments to the vector database and create an index. """
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db = Chroma.from_documents(documents=split_documents, embedding=embeddings, collection_name=collection_name)
        return db

    def search_vector_db(self, vector_store: Chroma, query: str, k: int = 5) -> list:
        """ Search the vector database for documents similar to the query. """
        embedding_vector = OpenAIEmbeddings(api_key=api_key).embed_query(query)
        return vector_store.similarity_search_by_vector_with_relevance_scores(embedding_vector, k=k)


