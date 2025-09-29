"""Document processing module using Pydantic models."""
from typing import List
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..models import DocumentMetadata, VectorSearchResult
from ..config import ConfigManager


class DocumentProcessor:
    """Enhanced document loader with Pydantic validation."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def load_document(self, metadata: DocumentMetadata):
        """Load a document based on its metadata."""
        if metadata.file_type == "csv":
            return self._load_csv(metadata.file_path)
        elif metadata.file_type == "pdf":
            return self._load_pdf(metadata.file_path)
        elif metadata.file_type == "text":
            return self._load_text(metadata.file_path)
        elif metadata.file_type == "web":
            return self._load_web(metadata.url)
        else:
            raise ValueError(f"Unsupported file type: {metadata.file_type}")
    
    def _load_csv(self, file_path: str) -> List:
        """Load a CSV file and return its contents."""
        loader = CSVLoader(file_path)
        return loader.load()
    
    def _load_pdf(self, file_path: str) -> List:
        """Load a PDF file and return its text."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_text(self, file_path: str) -> List:
        """Load a text file and return its contents."""
        loader = TextLoader(file_path)
        return loader.load()
    
    def _load_web(self, url: str) -> List:
        """Load a webpage and return its text content."""
        loader = WebBaseLoader(url)
        return loader.load()


class VectorStore:
    """Enhanced vector database with Pydantic validation."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def split_document(self, document, metadata: DocumentMetadata) -> List:
        """Split the document into smaller parts based on metadata settings."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=metadata.chunk_size, 
            chunk_overlap=metadata.chunk_overlap
        )
        return splitter.split_documents([document]) if not isinstance(document, list) else splitter.split_documents(document)
    
    def embed_upload(self, collection_name: str, split_documents: List) -> Chroma:
        """Upload document segments to the vector database and create an index."""
        api_key = self.config_manager.get_api_key("OpenAI")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db = Chroma.from_documents(
            documents=split_documents, 
            embedding=embeddings, 
            collection_name=collection_name
        )
        return db
    
    def search_vector_db(self, vector_store: Chroma, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Search the vector database for documents similar to the query."""
        api_key = self.config_manager.get_api_key("OpenAI")
        embedding_vector = OpenAIEmbeddings(api_key=api_key).embed_query(query)
        results = vector_store.similarity_search_by_vector_with_relevance_scores(embedding_vector, k=k)
        
        # Convert to Pydantic models
        return [
            VectorSearchResult(content=doc.page_content, score=score)
            for doc, score in results
        ]


def perform_vector_db_search(
    file_extension: str, 
    query: str, 
    config_manager: ConfigManager, 
    url: str = None, 
    k: int = 10
) -> List[VectorSearchResult]:
    """
    Perform a search in the vector database based on the file extension and query.
    """
    # Create metadata based on file extension
    if file_extension == "pdf":
        metadata = DocumentMetadata(file_type="pdf", file_path='./upload_docs/file.pdf')
    elif file_extension == "text":
        metadata = DocumentMetadata(file_type="text", file_path='./upload_docs/file.txt')
    elif file_extension == "csv":
        metadata = DocumentMetadata(file_type="csv", file_path='./upload_docs/file.csv')
    elif file_extension == "web":
        metadata = DocumentMetadata(file_type="web", url=url)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Process document
    processor = DocumentProcessor(config_manager)
    clean_text = processor.load_document(metadata)
    
    # Create vector store and search
    vector_store = VectorStore(config_manager)
    split_docs = vector_store.split_document(clean_text, metadata)
    db = vector_store.embed_upload("document_vector_db", split_docs)
    search_results = vector_store.search_vector_db(db, query, k)
    
    return search_results