import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
load_dotenv()

# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
# vector = embeddings.embed_query("hello, world!")
# vector[:5]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    api_key=os.environ["GOOGLE_GEMINI_API_KEY"],
    max_tokens=None,
    timeout=None,
    max_retries=2,

)




messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
# print(vector[:5])
