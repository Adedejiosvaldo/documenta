

# # import os
# # import re
# # from bs4 import BeautifulSoup
# # import asyncio
# # import aiohttp
# # from langchain_community.document_loaders import RecursiveUrlLoader
# # # import nest_asyncio
# # # nest_asyncio.apply()

# # from langchain_community.document_loaders.sitemap import SitemapLoader

# # # Define a BeautifulSoup extractor to parse HTML into cleaner text
# # def bs4_extractor(html: str) -> str:
# #     soup = BeautifulSoup(html, "lxml")
# #     return re.sub(r"\n\n+", "\n\n", soup.text).strip()



# # loader = RecursiveUrlLoader(
# #     "https://mongoosejs.com/docs/",
# #     # max_depth=2,
# #     use_async=True,
# #     extractor=bs4_extractor,
# #     # extractor=None,
# #     # metadata_extractor=None,
# #     # exclude_dirs=(),
# #     # timeout=10,
# #     # check_response_status=True,
# #     # continue_on_failure=True,
# #     # prevent_outside=True,
# #     # base_url=None,
# #     # ...
# # )


# # batch_size = 10
# # pages = []
# # processed_count = 0

# # print("Lazy loading documents in batches of", batch_size)
# # for doc in loader.lazy_load():
# #     pages.append(doc)
# #     processed_count += 1

# #     if len(pages) >= batch_size:
# #         # Process the current batch
# #         print(f"Processing batch of {len(pages)} documents (total processed: {processed_count})")

# #         # Example operation: print first 200 chars of the first document in batch
# #         if pages:
# #             print(f"Sample from current batch: {pages[0].page_content[:200]}...")

# #         # Here you would typically do something with the pages
# #         # For example: index.upsert(pages)

# #         # Clear the batch for the next iteration
# #         pages = []

# # # Process any remaining pages in the final batch
# # if pages:
# #     print(f"Processing final batch of {len(pages)} documents (total processed: {processed_count})")
# #     if pages:
# #         print(f"Sample from final batch: {pages[0].page_content[:200]}...")

# # print(pages[0].metadata)
# # # print(pages[3].page_content[1000:3000])




# import os
# import re
# from bs4 import BeautifulSoup
# import asyncio
# import aiohttp
# from typing import Union, Dict, Optional
# from langchain_community.document_loaders import RecursiveUrlLoader
# from langchain_community.document_transformers import Html2TextTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import time
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     filename='scraping.log'
# )
# logger = logging.getLogger('web_scraper')

# # Enhanced BeautifulSoup extractor with improved text cleanup
# def enhanced_bs4_extractor(html: str) -> str:
#     try:
#         soup = BeautifulSoup(html, "lxml")

#         # Remove script and style elements
#         for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
#             script_or_style.extract()

#         # Get text and clean it
#         text = soup.get_text(separator=' ', strip=True)

#         # Normalize whitespace
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'\n\n+', '\n\n', text).strip()

#         return text
#     except Exception as e:
#         logger.error(f"Error extracting text: {e}")
#         return ""

# # Advanced metadata extraction function
# def metadata_extractor(
#     raw_html: str,
#     url: str,
#     response: Union[aiohttp.ClientResponse, any]
# ) -> Dict:
#     """Extract useful metadata from HTML documents"""
#     metadata = {
#         "source": url,
#         "timestamp": time.time(),
#     }

#     # Extract response headers
#     if hasattr(response, "headers"):
#         metadata["content_type"] = response.headers.get("Content-Type", "")
#         metadata["last_modified"] = response.headers.get("Last-Modified", "")

#     # Extract more metadata using BeautifulSoup
#     try:
#         soup = BeautifulSoup(raw_html, "lxml")

#         # Get title
#         title_tag = soup.find("title")
#         if title_tag and title_tag.string:
#             metadata["title"] = title_tag.string.strip()

#         # Get meta description
#         desc_tag = soup.find("meta", attrs={"name": "description"})
#         if desc_tag and desc_tag.get("content"):
#             metadata["description"] = desc_tag.get("content").strip()

#         # Get language
#         html_tag = soup.find("html")
#         if html_tag and html_tag.get("lang"):
#             metadata["language"] = html_tag.get("lang")

#         # Get canonical URL
#         canonical = soup.find("link", attrs={"rel": "canonical"})
#         if canonical and canonical.get("href"):
#             metadata["canonical_url"] = canonical.get("href")

#     except Exception as e:
#         logger.error(f"Error extracting metadata: {e}")

#     return metadata

# # Custom headers to mimic a real browser
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#     "Accept-Language": "en-US,en;q=0.5",
#     "Connection": "keep-alive",
#     "Upgrade-Insecure-Requests": "1",
#     "Cache-Control": "max-age=0"
# }

# # Initialize the loader with optimized settings
# def create_optimized_loader(url: str, max_depth: int = 2, use_pattern: Optional[str] = None) -> RecursiveUrlLoader:
#     """Create an optimized RecursiveUrlLoader for a specific website"""

#     # Configure pattern matching for specific content if provided
#     link_pattern = None
#     if use_pattern:
#         link_pattern = use_pattern

#     return RecursiveUrlLoader(
#         url=url,
#         max_depth=max_depth,  # Control crawl depth
#         use_async=True,  # Use async for better performance
#         extractor=enhanced_bs4_extractor,  # Use enhanced text extraction
#         metadata_extractor=metadata_extractor,  # Extract rich metadata
#         exclude_dirs=("login", "signup", "cart", "checkout", "profile"),  # Skip irrelevant paths
#         timeout=15,  # Reasonable timeout
#         prevent_outside=True,  # Stay within the same site
#         link_regex=link_pattern,  # Optional pattern matching
#         headers=headers,  # Set custom headers
#         check_response_status=True,  # Skip error pages
#         continue_on_failure=True,  # Keep going if some pages fail
#         autoset_encoding=True,  # Handle encoding correctly
#     )

# # Process function with retries, rate limiting and error handling
# async def process_documents(loader, batch_size=10, save_directory="extracted_data", rate_limit_delay=1.0):
#     """Process documents with batching, rate limiting and error handling"""

#     # Create directory if not exists
#     os.makedirs(save_directory, exist_ok=True)

#     pages = []
#     processed_count = 0
#     start_time = time.time()

#     logger.info(f"Starting document extraction with batch size {batch_size}")

#     # Text splitter for very large documents
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=4000,
#         chunk_overlap=200,
#         length_function=len
#     )

#     # Process documents with async loading
#     try:
#         async for doc in loader.alazy_load():
#             # Rate limiting
#             await asyncio.sleep(rate_limit_delay)

#             pages.append(doc)
#             processed_count += 1

#             # Process in batches
#             if len(pages) >= batch_size:
#                 await process_batch(pages, processed_count, save_directory, splitter)
#                 pages = []  # Clear batch

#         # Process any remaining pages
#         if pages:
#             await process_batch(pages, processed_count, save_directory, splitter)

#         elapsed_time = time.time() - start_time
#         logger.info(f"Completed processing {processed_count} documents in {elapsed_time:.2f} seconds")
#         print(f"Successfully processed {processed_count} documents in {elapsed_time:.2f} seconds")

#     except Exception as e:
#         logger.error(f"Error in document processing: {e}")
#         print(f"Error during processing: {e}")

# async def process_batch(batch, total_processed, save_directory, splitter):
#     """Process a batch of documents"""
#     logger.info(f"Processing batch of {len(batch)} documents (total: {total_processed})")

#     for i, doc in enumerate(batch):
#         try:
#             # Handle large documents by splitting if needed
#             if len(doc.page_content) > 10000:  # Very large document
#                 chunks = splitter.split_documents([doc])
#                 # Process chunks as needed

#             # Save to file (example action)
#             filename = f"{save_directory}/doc_{total_processed-len(batch)+i+1}.txt"
#             with open(filename, 'w', encoding='utf-8') as f:
#                 # Save metadata as JSON header
#                 f.write(f"--- METADATA ---\n")
#                 for key, value in doc.metadata.items():
#                     f.write(f"{key}: {value}\n")
#                 f.write(f"--- CONTENT ---\n\n")
#                 f.write(doc.page_content)

#             # Print sample from first document in batch
#             if i == 0:
#                 print(f"Sample from current batch: {doc.page_content[:200]}...")

#         except Exception as e:
#             logger.error(f"Error processing document {total_processed-len(batch)+i+1}: {e}")

# # Main execution
# async def main():
#     # Target website URL
#     target_url = "https://mongoosejs.com/docs/"

#     # Create loader
#     loader = create_optimized_loader(
#         url=target_url,
#         max_depth=2  # Adjust based on site structure
#     )

#     # Process the documents
#     await process_documents(
#         loader=loader,
#         batch_size=10,
#         save_directory="extracted_mongoose_docs",
#         rate_limit_delay=0.5  # Adjust based on server limitations
#     )

# # Run the main function
# if __name__ == "__main__":
#     asyncio.run(main())










import os
import re
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from typing import Union, Dict, Optional
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import logging
import json
import getpass
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='scraping.log'
)
logger = logging.getLogger('web_scraper')

# Enhanced BeautifulSoup extractor with improved text cleanup
def enhanced_bs4_extractor(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
            script_or_style.extract()

        # Get text and clean it
        text = soup.get_text(separator=' ', strip=True)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text).strip()

        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

# Advanced metadata extraction function
def metadata_extractor(
    raw_html: str,
    url: str,
    response: Union[aiohttp.ClientResponse, any]
) -> Dict:
    """Extract useful metadata from HTML documents"""
    metadata = {
        "source": url,
        "timestamp": time.time(),
    }

    # Extract response headers
    if hasattr(response, "headers"):
        metadata["content_type"] = response.headers.get("Content-Type", "")
        metadata["last_modified"] = response.headers.get("Last-Modified", "")

    # Extract more metadata using BeautifulSoup
    try:
        soup = BeautifulSoup(raw_html, "lxml")

        # Get title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()

        # Get meta description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag.get("content").strip()

        # Get language
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            metadata["language"] = html_tag.get("lang")

        # Get canonical URL
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if canonical and canonical.get("href"):
            metadata["canonical_url"] = canonical.get("href")

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")

    return metadata

# Custom headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0"
}

# Initialize the loader with optimized settings
def create_optimized_loader(url: str, max_depth: int = 2, use_pattern: Optional[str] = None) -> RecursiveUrlLoader:
    """Create an optimized RecursiveUrlLoader for a specific website"""

    # Configure pattern matching for specific content if provided
    link_pattern = None
    if use_pattern:
        link_pattern = use_pattern

    return RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,  # Control crawl depth
        use_async=True,  # Use async for better performance
        extractor=enhanced_bs4_extractor,  # Use enhanced text extraction
        metadata_extractor=metadata_extractor,  # Extract rich metadata
        exclude_dirs=("login", "signup", "cart", "checkout", "profile"),  # Skip irrelevant paths
        timeout=15,  # Reasonable timeout
        prevent_outside=True,  # Stay within the same site
        link_regex=link_pattern,  # Optional pattern matching
        headers=headers,  # Set custom headers
        check_response_status=True,  # Skip error pages
        continue_on_failure=True,  # Keep going if some pages fail
        autoset_encoding=True,  # Handle encoding correctly
    )

# Process function with retries, rate limiting and error handling
async def process_documents(loader, batch_size=10, save_directory="extracted_data", rate_limit_delay=1.0):
    """Process documents with batching, rate limiting and error handling"""

    # Create directory if not exists
    os.makedirs(save_directory, exist_ok=True)

    pages = []
    processed_count = 0
    start_time = time.time()

    logger.info(f"Starting document extraction with batch size {batch_size}")

    # Text splitter for very large documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len
    )

    # Process documents with async loading
    try:
        async for doc in loader.alazy_load():
            # Rate limiting
            await asyncio.sleep(rate_limit_delay)

            pages.append(doc)
            processed_count += 1

            # Process in batches
            if len(pages) >= batch_size:
                await process_batch(pages, processed_count, save_directory, splitter)
                pages = []  # Clear batch

        # Process any remaining pages
        if pages:
            await process_batch(pages, processed_count, save_directory, splitter)

        elapsed_time = time.time() - start_time
        logger.info(f"Completed processing {processed_count} documents in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {processed_count} documents in {elapsed_time:.2f} seconds")

        return processed_count

    except Exception as e:
        logger.error(f"Error in document processing: {e}")
        print(f"Error during processing: {e}")
        return 0

async def process_batch(batch, total_processed, save_directory, splitter):
    """Process a batch of documents"""
    logger.info(f"Processing batch of {len(batch)} documents (total: {total_processed})")

    all_chunks = []

    for i, doc in enumerate(batch):
        try:
            # Handle large documents by splitting if needed
            if len(doc.page_content) > 10000:  # Very large document
                chunks = splitter.split_documents([doc])
                all_chunks.extend(chunks)
            else:
                all_chunks.append(doc)

            # Save to file (example action)
            filename = f"{save_directory}/doc_{total_processed-len(batch)+i+1}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                # Save metadata as JSON header
                f.write(f"--- METADATA ---\n")
                for key, value in doc.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"--- CONTENT ---\n\n")
                f.write(doc.page_content)

            # Print sample from first document in batch
            if i == 0:
                print(f"Sample from current batch: {doc.page_content[:200]}...")

        except Exception as e:
            logger.error(f"Error processing document {total_processed-len(batch)+i+1}: {e}")

    return all_chunks

# Function to load all documents from the saved files
def load_processed_documents(directory):
    """Load all processed documents from the directory"""
    documents = []

    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        return documents

    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

                # Parse the file content
                metadata = {}
                page_content = ""

                # Simple parsing of our custom format
                parts = content.split("--- CONTENT ---\n\n", 1)
                if len(parts) == 2:
                    metadata_section = parts[0].replace("--- METADATA ---\n", "")
                    page_content = parts[1]

                    # Parse metadata
                    for line in metadata_section.strip().split("\n"):
                        if ": " in line:
                            key, value = line.split(": ", 1)
                            metadata[key] = value

                from langchain_core.documents import Document
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")

    return documents

# Create vector store from documents
def create_vector_store(documents, embeddings_model=None):
    """Create a vector store from documents"""
    if not embeddings_model:
        embeddings_model = OpenAIEmbeddings()

    from langchain_core.vectorstores import InMemoryVectorStore

    print(f"Creating vector store with {len(documents)} documents")
    vector_store = InMemoryVectorStore.from_documents(
        documents,
        embeddings_model
    )
    return vector_store

# Create a RAG chain for querying documents
def create_rag_chain(vector_store):
    """Create a RAG chain for querying the vector store"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer the question in a comprehensive and informative way. If the information is not provided in the context, say "I don't have enough information to answer that question."
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Interactive chat function
def chat_with_documents(rag_chain):
    """Interactive chat with the documents"""
    print("\n=== Document Chat Interface ===")
    print("Type 'exit' to quit the chat")

    while True:
        query = input("\nQuestion: ")
        if query.lower() in ('exit', 'quit'):
            break

        try:
            response = rag_chain.invoke(query)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"Error: {e}")

# Function to save vector store for future use
def save_vector_store(vector_store, filepath):
    """Save the vector store for future use"""
    try:
        vector_store.save_local(filepath)
        print(f"Vector store saved to {filepath}")
    except AttributeError:
        logger.error("This vector store type doesn't support local saving")
        print("This vector store type doesn't support direct saving. Using FAISS instead.")

        # Convert to FAISS and save
        try:
            from langchain_community.vectorstores import FAISS

            # Convert the vector store to FAISS format
            docs = []
            for i in range(vector_store.index.shape[0]):
                doc = vector_store.docstore.search(str(i))
                docs.append(doc)

            faiss_store = FAISS.from_documents(docs, OpenAIEmbeddings())
            faiss_store.save_local(filepath)
            print(f"Vector store converted to FAISS and saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            print(f"Error saving vector store: {e}")

# Main execution
async def main():
    # Target website URL
    target_url = "https://mongoosejs.com/docs/"
    save_directory = "extracted_mongoose_docs"

    # Ask user if they want to scrape or use existing data
    if os.path.exists(save_directory) and len(os.listdir(save_directory)) > 0:
        choice = input(f"Found existing data in '{save_directory}'. Use it (y) or scrape again (n)? [y/n]: ")

        if choice.lower() != 'n':
            print("Using existing data...")
            documents = load_processed_documents(save_directory)
            print(f"Loaded {len(documents)} documents from {save_directory}")
        else:
            print("Scraping data again...")
            # Create loader
            loader = create_optimized_loader(
                url=target_url,
                max_depth=2  # Adjust based on site structure
            )

            # Process the documents
            await process_documents(
                loader=loader,
                batch_size=10,
                save_directory=save_directory,
                rate_limit_delay=0.5  # Adjust based on server limitations
            )
            documents = load_processed_documents(save_directory)
    else:
        print("No existing data found. Starting web scraping...")
        # Create loader
        loader = create_optimized_loader(
            url=target_url,
            max_depth=2  # Adjust based on site structure
        )

        # Process the documents
        await process_documents(
            loader=loader,
            batch_size=10,
            save_directory=save_directory,
            rate_limit_delay=0.5  # Adjust based on server limitations
        )
        documents = load_processed_documents(save_directory)

    # Create vector store
    vector_store = create_vector_store(documents)

    # Create RAG chain
    rag_chain = create_rag_chain(vector_store)

    # Demonstrate vector search
    print("\n=== Vector Search Example ===")
    query = "How to define a Mongoose schema"
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print(f"Query: {query}\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"Result {i+1} from {doc.metadata.get('source', 'unknown')}:")
        print(f"{doc.page_content[:300]}...\n")

    # Interactive chat session
    chat_with_documents(rag_chain)

    # Option to save vector store for future use
    save_choice = input("\nSave vector store for future use? [y/n]: ")
    if save_choice.lower() == 'y':
        save_vector_store(vector_store, "mongoose_docs_vectors")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
