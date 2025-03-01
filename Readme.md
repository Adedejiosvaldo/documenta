# Document AI

A tool for processing and querying documents using vector stores and chat services.

## Overview

- Loads documents from a designated folder.
- Stores documents in a vector store (using ChromaDB or Pinecone).
- Allows querying stored documents.
- Provides a chat interface for additional question answering.

## Features

- Document processing via DocumentProcessor.
- Vector storage with ChromaDB (alternative: Pinecone).
- Chat functionality using ChatService.

## Requirements

- Python 3.8+
- Required packages (install via pip).
  // ...existing instructions for dependencies...

## Installation

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your API keys and parameters in the environment or configuration files.

## Usage

Run the main script:

```
python src/main.py
```

Customize the code to switch between vector stores (ChromaDB or Pinecone) and set your API keys as needed.

## Code Structure

- `src/main.py`: Main entry point, initializes document processing, vector store, and chat service.
- `vector_store/`: Contains implementations for different vector stores.
- `chat/`: Contains chat service logic.

