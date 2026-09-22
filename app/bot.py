import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

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


# Для каждого пользователя храним текущий режим памяти.
# По умолчанию память включена.
memory_modes: dict[int, bool] = {}


def is_memory_enabled(user_id: int) -> bool:
    """Возвращает текущий режим памяти пользователя."""
    return memory_modes.get(user_id, True)


def build_memory_keyboard(
    memory_enabled: bool,
) -> InlineKeyboardMarkup:
    """Создаёт клавиатуру управления памятью."""

    if memory_enabled:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔴 Выключить память",
                        callback_data="memory_off",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🗑 Очистить память",
                        callback_data="memory_reset",
                    )
                ],
            ]
        )

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🧠 Включить память",
                    callback_data="memory_on",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Очистить память",
                    callback_data="memory_reset",
                )
            ],
        ]
    )


def memory_status_text(user_id: int) -> str:
    """Возвращает текст текущего состояния памяти."""

    if is_memory_enabled(user_id):
        return (
            "🧠 Память включена.\n\n"
            "Бот использует:\n"
            "• краткосрочную историю диалога;\n"
            "• долгосрочную память в ChromaDB.\n\n"
            "Сохранённые воспоминания доступны модели."
        )

    return (
        "🚫 Память выключена.\n\n"
        "Бот не использует:\n"
        "• историю текущего диалога;\n"
        "• долгосрочную память пользователя.\n\n"
        "Накопленные воспоминания не удалены."
    )


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    user_id = message.from_user.id

    await message.answer(
        "Привет! Я Telegram-бот с памятью.\n\n"
        "Я умею:\n"
        "• запоминать контекст диалога;\n"
        "• сохранять информацию между перезапусками;\n"
        "• искать информацию в загруженных документах;\n"
        "• полностью очищать личную память.\n\n"
        "Сейчас:\n"
        f"{memory_status_text(user_id)}\n\n"
        "Команда /memory открывает управление памятью."
    )


@dp.message(Command("memory"))
async def memory_handler(message: Message) -> None:
    user_id = message.from_user.id

    await message.answer(
        memory_status_text(user_id),
        reply_markup=build_memory_keyboard(
            is_memory_enabled(user_id)
        ),
    )


@dp.callback_query(F.data == "memory_on")
async def memory_on_callback(
    callback: CallbackQuery,
) -> None:
    user_id = callback.from_user.id

    memory_modes[user_id] = True

    await callback.answer("Память включена.")

    if callback.message:
        await callback.message.edit_text(
            memory_status_text(user_id),
            reply_markup=build_memory_keyboard(True),
        )


@dp.callback_query(F.data == "memory_off")
async def memory_off_callback(
    callback: CallbackQuery,
) -> None:
    user_id = callback.from_user.id

    memory_modes[user_id] = False

    await callback.answer("Память выключена.")

    if callback.message:
        await callback.message.edit_text(
            memory_status_text(user_id),
            reply_markup=build_memory_keyboard(False),
        )


@dp.callback_query(F.data == "memory_reset")
async def memory_reset_callback(
    callback: CallbackQuery,
) -> None:
    user_id = callback.from_user.id

    clear_all_memory(user_id)

    await callback.answer("Память очищена.")

    if callback.message:
        await callback.message.edit_text(
            "🗑 Память пользователя полностью очищена.\n\n"
            "Краткосрочная история и долгосрочная память "
            "удалены.\n\n"
            "База знаний загруженных документов не затронута.",
            reply_markup=build_memory_keyboard(
                is_memory_enabled(user_id)
            ),
        )


@dp.message(Command("reset"))
async def reset_handler(message: Message) -> None:
    user_id = message.from_user.id

    clear_all_memory(user_id)

    await message.answer(
        "🗑 Память очищена.\n\n"
        "Краткосрочная история и долгосрочная память "
        "удалены.\n"
        "База знаний документов не затронута."
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

    memory_enabled = is_memory_enabled(user_id)

    # --------------------------------------------------------
    # КРАТКОСРОЧНАЯ ПАМЯТЬ
    # --------------------------------------------------------

    if memory_enabled:
        add_message(
            user_id,
            "user",
            user_text,
        )

    # --------------------------------------------------------
    # EMBEDDING ТЕКУЩЕГО ЗАПРОСА
    # --------------------------------------------------------

    query_embedding = await create_embedding(user_text)

    # --------------------------------------------------------
    # ПОИСК ПО ДОКУМЕНТАМ
    #
    # Документы являются отдельной базой знаний.
    # Они доступны независимо от режима личной памяти.
    # --------------------------------------------------------

    relevant_chunks = search_memory(
        query_embedding
    )

    documents_context = "\n\n".join(
        relevant_chunks
    )

    # --------------------------------------------------------
    # ДОЛГОСРОЧНАЯ ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ
    # --------------------------------------------------------

    memory_context = ""

    if memory_enabled:
        remembered_messages = search_user_memory(
            user_id=user_id,
            embedding=query_embedding,
        )

        memory_context = "\n\n".join(
            remembered_messages
        )

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    if memory_enabled:
        system_prompt = (
            "Ты полезный Telegram-бот с памятью. "
            "Отвечай на русском языке.\n\n"
            "Используй историю текущего диалога "
            "для понимания контекста.\n\n"
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

    else:
        system_prompt = (
            "Ты полезный Telegram-бот. "
            "Отвечай на русском языке.\n\n"
            "Режим памяти выключен. "
            "Не используй и не запоминай личную историю "
            "пользователя.\n\n"
            "Обрабатывай текущий запрос независимо "
            "от предыдущих сообщений.\n\n"
            "Если есть информация из документов, "
            "используй её при ответе.\n\n"
            f"Информация из документов:\n"
            f"{documents_context or 'Нет подходящей информации.'}"
        )

    # --------------------------------------------------------
    # СООБЩЕНИЯ ДЛЯ LLM
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    if memory_enabled:
        messages.extend(
            get_history(user_id)
        )

    else:
        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    answer = await ask_llm(messages)

    # --------------------------------------------------------
    # СОХРАНЕНИЕ В ПАМЯТЬ
    # --------------------------------------------------------

    if memory_enabled:
        add_message(
            user_id,
            "assistant",
            answer,
        )

        save_user_memory(
            user_id=user_id,
            content=user_text,
            embedding=query_embedding,
        )

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