import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.config import TELEGRAM_BOT_TOKEN
from app.llm import ask_llm, create_embedding
from app.memory import (
    add_message,
    clear_all_memory,
    get_history,
    save_user_memory,
    search_memory,
    search_user_memory,
)


bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! Я бот с памятью.\n\n"
        "Я помню последние сообщения нашего диалога, "
        "сохраняю долгосрочную память и могу искать "
        "информацию в загруженных документах.\n\n"
        "Команда /reset полностью очищает мою память "
        "о текущем пользователе."
    )


@dp.message(Command("reset"))
async def reset_handler(message: Message) -> None:
    user_id = message.from_user.id

    clear_all_memory(user_id)

    await message.answer(
        "Память очищена. "
        "Начинаем диалог с чистого листа."
    )


@dp.message(F.document)
async def document_handler(message: Message) -> None:
    document = message.document

    if not document:
        return

    filename = document.file_name or "document"

    allowed_extensions = {
        ".txt",
        ".pdf",
        ".docx",
    }

    extension = (
        "." + filename.lower().split(".")[-1]
        if "." in filename
        else ""
    )

    if extension not in allowed_extensions:
        await message.answer(
            "Поддерживаются только файлы TXT, PDF и DOCX."
        )
        return

    from app.documents import UPLOADS_DIR, index_document

    file_path = UPLOADS_DIR / filename

    telegram_file = await bot.get_file(
        document.file_id
    )

    await bot.download_file(
        telegram_file.file_path,
        file_path,
    )

    try:
        chunks_count = await index_document(file_path)
    except Exception as error:
        await message.answer(
            f"Не удалось обработать файл: {error}"
        )
        return

    await message.answer(
        f"Файл «{filename}» обработан.\n"
        f"Добавлено фрагментов в память: {chunks_count}"
    )


@dp.message(F.text)
async def text_handler(message: Message) -> None:
    user_id = message.from_user.id
    user_text = message.text

    # --------------------------------------------------------
    # 1. Сохраняем сообщение в краткосрочную память
    # --------------------------------------------------------

    add_message(
        user_id,
        "user",
        user_text,
    )

    # --------------------------------------------------------
    # 2. Создаём embedding пользовательского сообщения
    # --------------------------------------------------------

    query_embedding = await create_embedding(
        user_text
    )

    # --------------------------------------------------------
    # 3. Ищем информацию в загруженных документах
    # --------------------------------------------------------

    relevant_chunks = search_memory(
        query_embedding
    )

    documents_context = "\n\n".join(
        relevant_chunks
    )

    # --------------------------------------------------------
    # 4. Ищем информацию в долгосрочной памяти
    # --------------------------------------------------------

    remembered_messages = search_user_memory(
        user_id=user_id,
        embedding=query_embedding,
    )

    memory_context = "\n\n".join(
        remembered_messages
    )

    # --------------------------------------------------------
    # 5. Формируем системный промпт
    # --------------------------------------------------------

    system_prompt = (
        "Ты полезный Telegram-бот с памятью. "
        "Отвечай на русском языке.\n\n"
        "Используй историю текущего диалога для понимания "
        "контекста.\n\n"
        "Если в долгосрочной памяти есть информация "
        "о пользователе или предыдущих сообщениях, "
        "используй её при ответе.\n\n"
        "Если есть информация из документов, "
        "используй её при ответе.\n\n"
        f"Долгосрочная память:\n"
        f"{memory_context or 'Нет сохранённой информации.'}\n\n"
        f"Информация из документов:\n"
        f"{documents_context or 'Нет подходящей информации.'}"
    )

    # --------------------------------------------------------
    # 6. Формируем запрос к LLM
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(
        get_history(user_id)
    )

    # --------------------------------------------------------
    # 7. Получаем ответ модели
    # --------------------------------------------------------

    answer = await ask_llm(messages)

    # --------------------------------------------------------
    # 8. Сохраняем ответ в краткосрочную память
    # --------------------------------------------------------

    add_message(
        user_id,
        "assistant",
        answer,
    )

    # --------------------------------------------------------
    # 9. Сохраняем пользовательское сообщение
    #    в долгосрочную память
    # --------------------------------------------------------

    save_user_memory(
        user_id=user_id,
        content=user_text,
        embedding=query_embedding,
    )

    # --------------------------------------------------------
    # 10. Сохраняем ответ бота
    #     в долгосрочную память
    # --------------------------------------------------------

    answer_embedding = await create_embedding(
        answer
    )

    save_user_memory(
        user_id=user_id,
        content=answer,
        embedding=answer_embedding,
    )

    await message.answer(answer)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())