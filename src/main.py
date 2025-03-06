

# main.py
import os
from vector_store import DocumentProcessor, ChromaStore
from chat import ChatService


def main():
    # Initialize document processor
    print("Hello")
    processor = DocumentProcessor(docs_dir="./src/scraped_docs")
    # documents = processor.load_documents()

# fOR faiss
    vector_store = FAISSVectorStore(processor=processor)
    vector_store.store_documents(documents)

    # For ChromaDB:
    vector_store = ChromaStore(db_path="./chroma_db", processor=processor)

    # Store documents
    # vector_store.store_documents(documents)

    # Initialize chat service
    chat_service = ChatService(vector_store=vector_store)

    # Example usage
    # Simple query
    question = "What are the schema type options in Mongoose?"
    # question = "How to define a schema in mongoose for a dog in typescript?"
    results = vector_store.query(question)
    print("Retrieved documents:")
    for doc in results:
        print(f"Source: {doc['source']}\nText: {doc['text'][:200]}...\n")

    # Chat example (if API key provided in .env)
    if chat_service.api_key:
        response = chat_service.chat(question)
        print(f"Answer: {response}")


if __name__ == "__main__":
    main()
