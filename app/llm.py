from openai import AsyncOpenAI

from app.config import (
    LLM_MODEL,
    PROXYAPI_BASE_URL,
    PROXYAPI_KEY,
)


client = AsyncOpenAI(
    api_key=PROXYAPI_KEY,
    base_url=PROXYAPI_BASE_URL,
)


async def ask_llm(messages: list[dict[str, str]]) -> str:
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )

    return response.choices[0].message.content or ""


async def create_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding