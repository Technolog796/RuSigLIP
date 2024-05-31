import json

import cv2
import numpy as np
from aiogram import Router
from aiogram.filters.command import Command
from aiogram.utils.formatting import as_marked_section
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
)

from database import get_language, set_language
from utils.inference_utils import load_model_and_tokenizer, preprocess, get_probs


model, tokenizer = load_model_and_tokenizer(
    model_weights="path/to/model/weights.safetensors"
)

with open("messages.json", "r", encoding="utf-8") as file:
    messages = json.load(file)

router = Router()


async def get_user_messages(user_id: int) -> dict[str, str]:
    language = await get_language(user_id)
    if language is not None:
        language = language[0]
    else:
        language = "en"
    return messages[language]


async def classify_image(image: np.ndarray, labels: list[str]):
    images, texts = preprocess([image], labels, tokenizer)
    img_emb, txt_emb = model(images, texts)
    probs = get_probs(img_emb, txt_emb)[0].tolist()

    combined = list(zip(labels, probs))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_probs = zip(*combined_sorted)
    return list(sorted_labels), list(sorted_probs)


@router.message(Command("start"))
async def start_handler(message: Message) -> None:
    start_message = messages["ru"]["start"] + "\n\n" + messages["en"]["start"]
    name_ru = message.from_user.full_name if message.from_user else "Пользователь"
    name_en = message.from_user.full_name if message.from_user else "User"
    await message.answer(start_message.format(name_ru, name_en))


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    user_id = message.from_user.id
    msgs = await get_user_messages(user_id)
    content = as_marked_section(*msgs["help"], marker="▶ ")
    await message.answer(**content.as_kwargs())


@router.message(Command("lang"))
async def settings_handler(message: Message) -> None:
    user_id = message.from_user.id
    msgs = await get_user_messages(user_id)

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="English", callback_data="lang_en"),
                InlineKeyboardButton(text="Русский", callback_data="lang_ru"),
            ]
        ]
    )
    await message.answer(msgs["choose_language"], reply_markup=keyboard)


@router.callback_query(lambda c: c.data.startswith("lang_"))
async def language_callback_handler(callback_query: CallbackQuery) -> None:
    user_id = callback_query.from_user.id
    data = callback_query.data

    if data == "lang_en":
        await set_language(user_id, "en")
        await callback_query.message.edit_text(messages["en"]["language_set"])
    elif data == "lang_ru":
        await set_language(user_id, "ru")
        await callback_query.message.edit_text(messages["ru"]["language_set"])

    await callback_query.answer()


@router.message()
async def process_media_and_text(message: Message) -> None:
    user_id = message.from_user.id
    msgs = await get_user_messages(user_id)

    image = None
    if message.photo:
        photo = message.photo[-1]
        file_info = await message.bot.get_file(photo.file_id)
        file_path = file_info.file_path
        file = await message.bot.download_file(file_path)
        img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        await message.answer(msgs["no_image_error"])
        return

    texts = []
    text = message.caption
    if text:
        texts = list(filter(lambda x: x != "", map(str.strip, text.split("\n"))))

    if len(texts) == 0:
        await message.answer(msgs["no_text_error"])
        return

    texts, probs = await classify_image(image, texts)

    msg = ""
    for i, (prob, text) in enumerate(zip(probs, texts)):
        msg += f'{i + 1}. "{text}": {prob}\n'

    await message.answer(msg)
