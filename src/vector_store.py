from abc import ABC, abstractmethod
import spacy
import chromadb
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from multiprocessing import Pool
from tqdm import tqdm
import os

# Global model
nlp = None


def init_model():
    global nlp
    nlp = spacy.load("en_core_web_md")


def embed_batch(texts: List[str]) -> List[List[float]]:
    global nlp
    return [nlp(text).vector.tolist() for text in texts]


def embed_batch_wrapper(texts):
    return embed_batch(texts)


class VectorStore(ABC):
    @abstractmethod
    def store_documents(self, documents: List[Dict]):
        pass

    @abstractmethod
    def query(self, question: str, k: int = 3) -> List[Dict]:
        pass


class DocumentProcessor:
    def __init__(self, docs_dir: str, n_process=4):
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.n_process = min(n_process, os.cpu_count())
        init_model()  # Load spaCy in the main process

    def load_documents(self) -> List[Dict]:
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

    def embed(self, text: str) -> List[float]:
        return nlp(text).vector.tolist()


class ChromaStore(VectorStore):
    def __init__(self, db_path: str, processor: DocumentProcessor):
        self.client = chromadb.PersistentClient(path=db_path)
        self.client.delete_collection("docs_collection")
        self.collection = self.client.create_collection(name="docs_collection")
        self.processor = processor

    def store_documents(self, documents: List[Dict]):
        batch_size = 500
        print(f"Embedding and storing {len(documents)} documents...")

        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        text_batches = [[doc["text"] for doc in batch] for batch in batches]

        with Pool(processes=self.processor.n_process, initializer=init_model) as pool:
            results = list(
                tqdm(
                    pool.imap(embed_batch_wrapper, text_batches),
                    total=len(batches),
                    desc="Embedding",
                )
            )

        for i, (batch, embeddings) in enumerate(zip(batches, results)):
            self.collection.add(
                documents=[doc["text"] for doc in batch],
                embeddings=embeddings,
                metadatas=[doc["metadata"] for doc in batch],
                ids=[doc["id"] for doc in batch],
            )
            print(f"Stored batch {i + 1}/{len(batches)}")

    def query(self, question: str, k: int = 3) -> List[Dict]:
        query_embedding = self.processor.embed(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return [
            {"text": text, "source": metadata["source"]}
            for text, metadata in zip(results["documents"][0], results["metadatas"][0])
        ]
