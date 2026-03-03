import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from logger import get_logger

log = get_logger(__name__)

log.info("Loading environment variables from .env ...")
loaded = load_dotenv()
if loaded:
    log.info(".env file found and loaded")
else:
    log.info("No .env file found, using system environment")

MODEL_NAME = os.getenv("LLM_MODEL")
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

missing = [k for k, v in {"LLM_MODEL": MODEL_NAME, "LLM_API_KEY": API_KEY, "LLM_BASE_URL": BASE_URL}.items() if not v]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

log.info("Environment OK | LLM_MODEL=%s  LLM_BASE_URL=%s  LLM_API_KEY=***%s",
         MODEL_NAME, BASE_URL, (API_KEY or "")[-4:])

log.info("Initialising LLM client ...")
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
)
log.info("LLM client ready")
