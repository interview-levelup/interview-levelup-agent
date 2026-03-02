import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL")
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

if not MODEL_NAME or not API_KEY or not BASE_URL:
    raise RuntimeError("LLM_MODEL, LLM_API_KEY and LLM_BASE_URL must be set in environment")

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
)
