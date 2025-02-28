# # # vector_store.py
# # from abc import ABC, abstractmethod
# # from sentence_transformers import SentenceTransformer
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import pinecone
# # import chromadb
# # from chromadb.utils import embedding_functions
# # from typing import List, Dict
# # import os


# # class VectorStore(ABC):
# #     @abstractmethod
# #     def store_documents(self, documents: List[Dict]):
# #         pass

# #     @abstractmethod
# #     def query(self, question: str, k: int = 3) -> List[Dict]:
# #         pass


# # class DocumentProcessor:
# #     def __init__(self, docs_dir: str):
# #         self.docs_dir = docs_dir
# #         self.model = SentenceTransformer("all-MiniLM-L6-v2")
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=1000, chunk_overlap=200, length_function=len
# #         )

# #     def load_documents(self) -> List[Dict]:
# #         """Load and process all text files from the directory"""
# #         documents = []
# #         for filename in os.listdir(self.docs_dir):
# #             if filename.endswith(".txt"):
# #                 filepath = os.path.join(self.docs_dir, filename)
# #                 with open(filepath, "r", encoding="utf-8") as f:
# #                     text = f.read()
# #                     chunks = self.text_splitter.split_text(text)
# #                     for i, chunk in enumerate(chunks):
# #                         documents.append(
# #                             {
# #                                 "id": f"{filename}_{i}",
# #                                 "text": chunk,
# #                                 "metadata": {"source": filename},
# #                             }
# #                         )
# #         return documents


# # class PineconeStore(VectorStore):
# #     def __init__(self, api_key: str, index_name: str):
# #         pinecone.init(api_key=api_key, environment="us-west1-gcp")
# #         self.index = pinecone.Index(index_name)
# #         self.model = SentenceTransformer("all-MiniLM-L6-v2")

# #     def store_documents(self, documents: List[Dict]):
# #         batch_size = 100
# #         for i in range(0, len(documents), batch_size):
# #             batch = documents[i : i + batch_size]
# #             embeddings = self.model.encode([doc["text"] for doc in batch])
# #             vectors = [
# #                 (doc["id"], embedding.tolist(), doc["metadata"])
# #                 for doc, embedding in zip(batch, embeddings)
# #             ]
# #             self.index.upsert(vectors=vectors)
# #             print(f"Stored batch {i//batch_size + 1}")

# #     def query(self, question: str, k: int = 3) -> List[Dict]:
# #         query_embedding = self.model.encode(question).tolist()
# #         results = self.index.query(
# #             vector=query_embedding, top_k=k, include_metadata=True
# #         )
# #         return [
# #             {
# #                 "text": match["metadata"].get("text", ""),
# #                 "source": match["metadata"]["source"],
# #             }
# #             for match in results["matches"]
# #         ]


# # class ChromaStore(VectorStore):
# #     def __init__(self, db_path: str = "./chroma_db"):
# #         self.client = chromadb.PersistentClient(path=db_path)
# #         self.collection = self.client.get_or_create_collection(
# #             name="docs_collection",
# #             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
# #                 model_name="all-MiniLM-L6-v2"
# #             ),
# #         )

# #     def store_documents(self, documents: List[Dict]):
# #         self.collection.add(
# #             documents=[doc["text"] for doc in documents],
# #             metadatas=[doc["metadata"] for doc in documents],
# #             ids=[doc["id"] for doc in documents],
# #         )
# #         print("Stored all documents in Chroma")

# #     def query(self, question: str, k: int = 3) -> List[Dict]:
# #         results = self.collection.query(query_texts=[question], n_results=k)
# #         return [
# #             {"text": text, "source": metadata["source"]}
# #             for text, metadata in zip(results["documents"][0], results["metadatas"][0])
# #         ]


# from abc import ABC, abstractmethod
# import spacy
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pinecone
# import chromadb
# from typing import List, Dict
# import os


# class VectorStore(ABC):
#     @abstractmethod
#     def store_documents(self, documents: List[Dict]):
#         pass

#     @abstractmethod
#     def query(self, question: str, k: int = 3) -> List[Dict]:
#         pass


# class DocumentProcessor:
#     def __init__(self, docs_dir: str):
#         self.docs_dir = docs_dir
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, length_function=len
#         )
#         self.nlp = spacy.load("en_core_web_md")

#     def load_documents(self) -> List[Dict]:
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
#         return documents

#     def embed(self, text: str):
#         return self.nlp(text).vector


# class PineconeStore(VectorStore):
#     def __init__(self, api_key: str, index_name: str, processor: DocumentProcessor):
#         pinecone.init(api_key=api_key, environment="us-west1-gcp")
#         self.index = pinecone.Index(index_name)
#         self.processor = processor

#     def store_documents(self, documents: List[Dict]):
#         batch_size = 100
#         for i in range(0, len(documents), batch_size):
#             batch = documents[i : i + batch_size]
#             embeddings = [self.processor.embed(doc["text"]).tolist() for doc in batch]
#             vectors = [
#                 (doc["id"], embedding, doc["metadata"])
#                 for doc, embedding in zip(batch, embeddings)
#             ]
#             self.index.upsert(vectors=vectors)
#             print(f"Stored batch {i // batch_size + 1}")

#     def query(self, question: str, k: int = 3) -> List[Dict]:
#         query_embedding = self.processor.embed(question).tolist()
#         results = self.index.query(
#             vector=query_embedding, top_k=k, include_metadata=True
#         )
#         return [
#             {"text": match["metadata"]["source"], "source": match["metadata"]["source"]}
#             for match in results["matches"]
#         ]


# vector_store.py
from abc import ABC, abstractmethod
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import chromadb
from typing import List, Dict
import os


class VectorStore(ABC):
    @abstractmethod
    def store_documents(self, documents: List[Dict]):
        pass

    @abstractmethod
    def query(self, question: str, k: int = 3) -> List[Dict]:
        pass


class DocumentProcessor:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.nlp = spacy.load("en_core_web_md")

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
        return documents

    def embed(self, text: str) -> List[float]:
        return self.nlp(text).vector.tolist()


class PineconeStore(VectorStore):
    def __init__(self, api_key: str, index_name: str, processor: DocumentProcessor):
        pinecone.init(api_key=api_key, environment="us-west1-gcp")
        self.index = pinecone.Index(index_name)
        self.processor = processor

    def store_documents(self, documents: List[Dict]):
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = [self.processor.embed(doc["text"]) for doc in batch]
            vectors = [
                (
                    doc["id"],
                    embedding,
                    {"source": doc["metadata"]["source"], "text": doc["text"]},
                )
                for doc, embedding in zip(batch, embeddings)
            ]
            self.index.upsert(vectors=vectors)
            print(f"Stored batch {i // batch_size + 1}")

    def query(self, question: str, k: int = 3) -> List[Dict]:
        query_embedding = self.processor.embed(question)
        results = self.index.query(
            vector=query_embedding, top_k=k, include_metadata=True
        )
        return [
            {"text": match["metadata"]["text"], "source": match["metadata"]["source"]}
            for match in results["matches"]
        ]


class ChromaStore(VectorStore):
    def __init__(self, db_path: str, processor: DocumentProcessor):
        self.client = chromadb.PersistentClient(path=db_path)
        # Reset the collection to avoid duplicates
        self.client.delete_collection("docs_collection")
        self.collection = self.client.create_collection(name="docs_collection")
        self.processor = processor

    def store_documents(self, documents: List[Dict]):
        embeddings = [self.processor.embed(doc["text"]) for doc in documents]
        existing_ids = set(self.collection.get()["ids"])
        new_docs = [doc for doc in documents if doc["id"] not in existing_ids]

        if new_docs:
            self.collection.add(
                documents=[doc["text"] for doc in new_docs],
                embeddings=[self.processor.embed(doc["text"]) for doc in new_docs],
                metadatas=[doc["metadata"] for doc in new_docs],
                ids=[doc["id"] for doc in new_docs],
            )
            print(f"Stored {len(new_docs)} new documents in Chroma")
        else:
            print("No new documents to store")

    def query(self, question: str, k: int = 3) -> List[Dict]:
        query_embedding = self.processor.embed(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return [
            {"text": text, "source": metadata["source"]}
            for text, metadata in zip(results["documents"][0], results["metadatas"][0])
        ]


# class ChromaStore(VectorStore):
#     def __init__(self, db_path: str, processor: DocumentProcessor):
#         self.client = chromadb.PersistentClient(path=db_path)
#         self.collection = self.client.get_or_create_collection(name="docs_collection")
#         self.processor = processor

#     def store_documents(self, documents: List[Dict]):
#         embeddings = [self.processor.embed(doc["text"]) for doc in documents]
#         self.collection.add(
#             documents=[doc["text"] for doc in documents],
#             embeddings=embeddings,
#             metadatas=[doc["metadata"] for doc in documents],
#             ids=[doc["id"] for doc in documents],
#         )
#         print("Stored all documents in Chroma")

#     def query(self, question: str, k: int = 3) -> List[Dict]:
#         query_embedding = self.processor.embed(question)
#         results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
#         return [
#             {"text": text, "source": metadata["source"]}
#             for text, metadata in zip(results["documents"][0], results["metadatas"][0])
#         ]
