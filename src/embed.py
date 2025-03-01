from langchain_huggingface.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector = embeddings.embed_query("Hello world!")
print(vector[:5])
