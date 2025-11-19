"""Document processing module using Pydantic models with enhanced RAG capabilities."""
from typing import List, Dict, Optional, Any
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

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


class EnhancedTextSplitter:
    """Factory for creating different types of text splitters."""

    @staticmethod
    def get_splitter(
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        **kwargs
    ):
        """
        Get a text splitter based on strategy.

        Args:
            strategy: Splitting strategy ('recursive', 'token', 'character', 'markdown', 'code')
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            **kwargs: Additional parameters for specific splitters

        Returns:
            Text splitter instance
        """
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=kwargs.get("separators", ["\n\n", "\n", ". ", " ", ""])
            )
        elif strategy == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif strategy == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=kwargs.get("separator", "\n\n")
            )
        elif strategy == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif strategy == "code":
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )


class VectorStore:
    """Enhanced vector database with advanced RAG capabilities."""

    def __init__(self, config_manager: ConfigManager, persist_directory: Optional[str] = None):
        """
        Initialize vector store.

        Args:
            config_manager: Configuration manager
            persist_directory: Optional directory to persist the vector store
        """
        self.config_manager = config_manager
        self.persist_directory = persist_directory

    def split_document(
        self,
        document,
        metadata: DocumentMetadata,
        strategy: str = "recursive",
        add_metadata: bool = True
    ) -> List[Document]:
        """
        Split documents using various strategies with enhanced metadata.

        Args:
            document: Document(s) to split
            metadata: Document metadata
            strategy: Splitting strategy
            add_metadata: Whether to add metadata to chunks

        Returns:
            List of split documents with metadata
        """
        splitter = EnhancedTextSplitter.get_splitter(
            strategy=strategy,
            chunk_size=metadata.chunk_size,
            chunk_overlap=metadata.chunk_overlap
        )

        documents = [document] if not isinstance(document, list) else document
        split_docs = splitter.split_documents(documents)

        # Add enhanced metadata to each chunk
        if add_metadata:
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(doc.page_content),
                    "file_type": metadata.file_type,
                    "source": metadata.file_path or metadata.url,
                    "splitting_strategy": strategy
                })

        return split_docs

    def embed_upload(
        self,
        collection_name: str,
        split_documents: List[Document],
        embedding_model: str = "text-embedding-3-small"
    ) -> Chroma:
        """
        Upload document segments to vector database with advanced embeddings.

        Args:
            collection_name: Name of the collection
            split_documents: Documents to embed
            embedding_model: OpenAI embedding model to use

        Returns:
            Chroma vector store instance
        """
        api_key = self.config_manager.get_api_key("OpenAI")
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model
        )

        if self.persist_directory:
            db = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
        else:
            db = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                collection_name=collection_name
            )

        return db

    def search_vector_db(
        self,
        vector_store: Chroma,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity"
    ) -> List[VectorSearchResult]:
        """
        Search vector database with advanced filtering and search types.

        Args:
            vector_store: Chroma vector store
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters to apply
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')

        Returns:
            List of VectorSearchResult with relevance scores
        """
        api_key = self.config_manager.get_api_key("OpenAI")

        if search_type == "mmr":
            # Maximal Marginal Relevance search (diversity)
            results = vector_store.max_marginal_relevance_search(
                query,
                k=k,
                filter=filter_dict
            )
            # MMR doesn't return scores, so we assign default
            return [
                VectorSearchResult(content=doc.page_content, score=1.0 - (i * 0.1))
                for i, doc in enumerate(results)
            ]
        else:
            # Similarity search with scores
            embedding_vector = OpenAIEmbeddings(api_key=api_key).embed_query(query)
            results = vector_store.similarity_search_by_vector_with_relevance_scores(
                embedding_vector,
                k=k,
                filter=filter_dict
            )

            return [
                VectorSearchResult(content=doc.page_content, score=score)
                for doc, score in results
            ]

    def hybrid_search(
        self,
        vector_store: Chroma,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            vector_store: Chroma vector store
            query: Search query
            k: Number of results
            filter_dict: Metadata filters

        Returns:
            List of VectorSearchResult combining both search methods
        """
        # Get semantic search results
        semantic_results = self.search_vector_db(
            vector_store, query, k=k, filter_dict=filter_dict
        )

        # Get MMR results for diversity
        mmr_results = self.search_vector_db(
            vector_store, query, k=k, filter_dict=filter_dict, search_type="mmr"
        )

        # Combine and deduplicate results
        seen = set()
        combined = []

        for result in semantic_results + mmr_results:
            if result.content not in seen:
                seen.add(result.content)
                combined.append(result)

        # Return top k unique results
        return combined[:k]

    def get_relevant_context(
        self,
        vector_store: Chroma,
        query: str,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Get relevant context formatted for RAG, respecting token limits.

        Args:
            vector_store: Chroma vector store
            query: Search query
            max_tokens: Maximum tokens for context
            **kwargs: Additional search parameters

        Returns:
            Formatted context string
        """
        results = self.search_vector_db(vector_store, query, **kwargs)

        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            chunk_text = f"[Document {i}] (Relevance: {result.score:.2f})\n{result.content}\n"
            chunk_length = len(chunk_text.split())  # Rough token estimate

            if current_length + chunk_length <= max_tokens:
                context_parts.append(chunk_text)
                current_length += chunk_length
            else:
                break

        return "\n".join(context_parts)


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