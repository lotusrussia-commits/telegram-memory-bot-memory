import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PROXYAPI_KEY = os.getenv("PROXYAPI_KEY")
PROXYAPI_BASE_URL = os.getenv(
    "PROXYAPI_BASE_URL",
    "https://api.proxyapi.ru/openai/v1",
)
LLM_MODEL = os.getenv("LLM_MODEL")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не задан TELEGRAM_BOT_TOKEN")

if not PROXYAPI_KEY:
    raise ValueError("Не задан PROXYAPI_KEY")

if not LLM_MODEL:
    raise ValueError("Не задан LLM_MODEL")