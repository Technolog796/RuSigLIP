from aiosqlite import Row, connect


async def start_database() -> None:
    sql_create_languages = """
        CREATE TABLE IF NOT EXISTS Languages (
           user_id INTEGER PRIMARY KEY,
           lang TEXT
        );
    """
    async with connect("database.db") as database:
        await database.execute(sql_create_languages)
        await database.commit()


async def get_language(user_id: int) -> Row | None:
    sql_select_language = """
        SELECT lang
        FROM Languages
        WHERE user_id = ?;
    """
    async with connect("database.db") as database:
        async with database.execute(sql_select_language, (user_id,)) as cursor:
            return await cursor.fetchone()


async def set_language(user_id: int, lang: str) -> None:
    sql_insert_language = """
        INSERT INTO Languages (user_id, lang)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET lang=excluded.lang;
    """
    async with connect("database.db") as database:
        await database.execute(sql_insert_language, (user_id, lang))
        await database.commit()
