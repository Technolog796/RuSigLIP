import asyncio
import logging
import configparser

from aiogram import Bot, Dispatcher

from database import start_database
from handlers import router


logging.basicConfig(level=logging.INFO)


async def main() -> None:
    await start_database()

    config = configparser.ConfigParser()
    config.read("settings.ini")
    bot = Bot(token=config["Telegram"]["TOKEN"])
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
