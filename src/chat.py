# # chat.py
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from typing import List, Dict
# from dotenv import load_dotenv

# import os


# class ChatService:
#     def __init__(self, vector_store: "VectorStore", api_key: str = None):
#         self.vector_store = vector_store
#         self.api_key = api_key
#         self.llm = None
#         if api_key:
#             self.llm = ChatGoogleGenerativeAI(
#                 model="gemini-pro",
#                 google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
#                 temperature=0.1,
#             )

#     def get_context(self, question: str, k: int = 3) -> str:
#         """Retrieve relevant context from vector store"""
#         relevant_docs = self.vector_store.query(question, k)
#         return "\n\n".join([doc["text"] for doc in relevant_docs])

#     def chat(self, question: str, k: int = 3) -> str:
#         """Generate a response using retrieved documents and Google Gemini"""
#         if not self.api_key:
#             raise ValueError("Google API key required for chat functionality")

#         context = self.get_context(question, k)
#         prompt_template = """Using the following documentation context, answer the question:

# Context:
# {context}

# Question: {question}
# Answer: """

#         prompt = PromptTemplate(
#             input_variables=["context", "question"], template=prompt_template
#         )

#         chain = prompt.format(context=context, question=question)
#         response = self.llm.invoke(chain)

#         return response.content


# chat.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Dict
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class ChatService:
    def __init__(self, vector_store: "FAISSVectorStore", api_key: str = None):
    # def __init__(self, vector_store: "VectorStore", api_key: str = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        self.llm = None
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key,
                temperature=0.1,
            )



    def chat(self, question: str, k: int = 3) -> str:
        """Generate a response using retrieved documents and Google Gemini."""
        if not self.api_key:
            raise ValueError("Google Gemini API key required for chat functionality")

        # Retrieve context from vector store
        relevant_docs = self.vector_store.query(question, k)
        context = "\n\n".join([doc["text"] for doc in relevant_docs])

        # Generate response
        prompt_template = """Using the following documentation context, answer the question:

Context:
{context}

Question: {question}
Answer: """
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )
        chain = prompt.format(context=context, question=question)
        response = self.llm.invoke(chain)
        return response.content









    # def get_context(self, question: str, k: int = 3) -> str:
    #     """Retrieve relevant context from vector store"""
    #     relevant_docs = self.vector_store.query(question, k)
    #     context = "\n\n".join([doc["text"] for doc in relevant_docs])
    #     print(f"Retrieved context: {context[:500]}...")  # Debug output
    #     return context




#     def chat(self, question: str, k: int = 3) -> str:
#         """Generate a response using retrieved documents and Google Gemini"""
#         if not self.api_key:
#             raise ValueError("Google Gemini API key required for chat functionality")

#         if not self.llm:
#             raise ValueError("LLM not initialized due to missing API key")

#         context = self.get_context(question, k)
#         prompt_template = """Using the following documentation context, answer the question:

# Context:
# {context}

# Question: {question}
# Answer: """

#         prompt = PromptTemplate(
#             input_variables=["context", "question"], template=prompt_template
#         )

#         chain = prompt.format(context=context, question=question)
#         response = self.llm.invoke(chain)
#         return response.content
