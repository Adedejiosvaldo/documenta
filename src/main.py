# # main.py
# from vector_store import DocumentProcessor, PineconeStore, ChromaStore
# from chat import ChatService


# def main():
#     # Initialize document processor
#     processor = DocumentProcessor(docs_dir="scraped_docs")
#     documents = processor.load_documents()

#     # Choose vector store
#     # For Pinecone:
#     """
#     vector_store = PineconeStore(
#         api_key="YOUR_PINECONE_API_KEY",
#         index_name="docs-index"
#     )
#     """

#     # For ChromaDB:
#     vector_store = ChromaStore(db_path="./chroma_db")

#     # Store documents
#     vector_store.store_documents(documents)

#     # Initialize chat service
#     chat_service = ChatService(
#         vector_store=vector_store,
#         api_key="YOUR_OPENAI_API_KEY",  # Optional, only if using chat
#     )

#     # Example usage
#     # Simple query
#     question = "What is asyncio in Python?"
#     results = vector_store.query(question)
#     print("Retrieved documents:")
#     for doc in results:
#         print(f"Source: {doc['source']}\nText: {doc['text'][:200]}...\n")

#     # Chat example (if API key provided)
#     if chat_service.api_key:
#         response = chat_service.chat(question)
#         print(f"Answer: {response}")


# if __name__ == "__main__":
#     main()


# main.py
import os
from vector_store import DocumentProcessor, ChromaStore
from chat import ChatService


def main():
    # Initialize document processor
    print("Hello")
    processor = DocumentProcessor(docs_dir="./src/scraped_docs")
    documents = processor.load_documents()

    # Choose vector store
    # For Pinecone:
    # vector_store = PineconeStore(
    #     api_key="YOUR_PINECONE_API_KEY", index_name="docs-index", processor=processor
    # )

    # For ChromaDB:
    vector_store = ChromaStore(db_path="./chroma_db", processor=processor)

    # Store documents
    vector_store.store_documents(documents)

    # Initialize chat service
    chat_service = ChatService(vector_store=vector_store)

    # Example usage
    # Simple query
    question = "How to define a schema in mongoose?"
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
