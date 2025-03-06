
from abc import ABC, abstractmethod
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from multiprocessing import Pool
from tqdm import tqdm
import os


# Initialize LangChain Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


class VectorStore(ABC):
    @abstractmethod
    def store_documents(self, documents: List[Dict]):
        pass

    @abstractmethod
    def query(self, question: str, k: int = 3) -> List[Dict]:
        pass


# class DocumentProcessor:
#     def __init__(self, docs_dir: str, n_process=4):
#         self.docs_dir = docs_dir
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, length_function=len
#         )
#         self.n_process = min(n_process, os.cpu_count() or 4)  # Safe fallback

#     def load_documents(self) -> List[Dict]:
#         """Load documents from directory and split them into chunks."""
#         documents = []
#         for filename in os.listdir(self.docs_dir):
#             if filename.endswith(".txt"):
#                 filepath = os.path.join(self.docs_dir, filename)
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     text = f.read()
#                     chunks = self.text_splitter.split_text(text)
#                     for i, chunk in enumerate(chunks):
#                         documents.append(
#                             {
#                                 "id": f"{filename}_{i}",
#                                 "text": chunk,
#                                 "metadata": {"source": filename},
#                             }
#                         )
#         print(f"Loaded {len(documents)} document chunks")
#         return documents

#     def embed(self, text: str) -> List[float]:
#         """Embed a single text chunk using LangChain's HuggingFaceEmbeddings."""
#         return embeddings.embed_query(text)



class DocumentProcessor:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

    def load_documents(self) -> List[Dict]:
        """Load documents from directory and split them into chunks."""
        documents = []
        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = self.text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        documents.append(
                            {
                                "id": f"{filename}_{i}",
                                "text": chunk,
                                "metadata": {"source": filename},
                            }
                        )
        print(f"Loaded {len(documents)} document chunks")
        return documents




class ChromaStore(VectorStore):
    def __init__(self, db_path: str, processor: DocumentProcessor):
        self.client = chromadb.PersistentClient(path=db_path)
        # self.client.delete_collection("docs_collection")
        self.collection = self.client.get_or_create_collection(name="docs_collection")
        self.processor = processor

    def store_documents(self, documents: List[Dict]):
        """Embed and store documents in batches using LangChain."""
        batch_size = 200  # Lower batch size for laptops
        print(f"Embedding and storing {len(documents)} documents...")

        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]
            embeddings_list = [self.processor.embed(text) for text in texts]

            self.collection.add(
                documents=texts,
                embeddings=embeddings_list,
                metadatas=[doc["metadata"] for doc in batch],
                ids=[doc["id"] for doc in batch],
            )
            print(f"Stored batch {i // batch_size + 1}")

    def query(self, question: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant documents for a query."""
        query_embedding = self.processor.embed(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        return [
            {"text": text, "source": metadata["source"]}
            for text, metadata in zip(results["documents"][0], results["metadatas"][0])
        ]
