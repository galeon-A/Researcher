"""
vectorstore/faiss_manager.py
Manages a FAISS vector store backed by Google Generative AI embeddings.
Supports building, saving, loading, and querying the store.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


VECTORSTORE_DIR = Path("vectorstore/data")
VECTORSTORE_INDEX = VECTORSTORE_DIR / "faiss_index"


class FAISSManager:
    """
    Manages FAISS vector store with Google AI embeddings.
    Persists to disk so repeated searches reuse cached embeddings.
    """

    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
        )
        self.vectorstore: Optional[FAISS] = None
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    def build_from_documents(self, raw_docs: list[dict], topic: str) -> FAISS:
        """
        Build a FAISS index from a list of raw document dicts.
        Each dict has 'content' and 'metadata' keys.
        """
        documents = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in raw_docs
        ]

        # Chunk long documents for better retrieval
        chunked_docs = self._chunk_documents(documents)

        self.vectorstore = FAISS.from_documents(chunked_docs, self.embeddings)
        self._save(topic)
        return self.vectorstore

    def _chunk_documents(
        self, documents: list[Document], chunk_size: int = 800, overlap: int = 100
    ) -> list[Document]:
        """Split long documents into overlapping chunks."""
        chunked = []
        for doc in documents:
            text = doc.page_content
            if len(text) <= chunk_size:
                chunked.append(doc)
            else:
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end]
                    chunked.append(
                        Document(page_content=chunk_text, metadata=doc.metadata)
                    )
                    start += chunk_size - overlap
        return chunked

    def similarity_search(self, query: str, k: int = 6) -> list[Document]:
        """Search the vector store for relevant chunks."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_from_documents first.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 6):
        """Return a LangChain-compatible retriever."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def format_context(self, query: str, k: int = 6) -> str:
        """Retrieve and format context as a single string for agent prompts."""
        docs = self.similarity_search(query, k=k)
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            ref = f"[{i}] {meta.get('title', 'Unknown')} ({meta.get('published', 'N/A')})"
            context_parts.append(f"{ref}\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)

    def get_all_references(self) -> list[dict]:
        """Return unique paper references from the vector store."""
        if self.vectorstore is None:
            return []
        # Access internal docstore
        docs = list(self.vectorstore.docstore._dict.values())
        seen = set()
        refs = []
        for doc in docs:
            meta = doc.metadata
            key = meta.get("arxiv_id", meta.get("title", ""))
            if key and key not in seen:
                seen.add(key)
                refs.append(meta)
        return refs

    def _save(self, topic: str):
        """Save the FAISS index to disk."""
        if self.vectorstore:
            save_path = str(VECTORSTORE_INDEX) + f"_{self._safe_name(topic)}"
            self.vectorstore.save_local(save_path)

    def load(self, topic: str) -> bool:
        """Load a previously saved FAISS index. Returns True if successful."""
        load_path = str(VECTORSTORE_INDEX) + f"_{self._safe_name(topic)}"
        if os.path.exists(load_path):
            try:
                self.vectorstore = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def _safe_name(topic: str) -> str:
        """Convert topic to a safe filename."""
        return "".join(c if c.isalnum() else "_" for c in topic.lower())[:50]
