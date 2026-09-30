"""
Girl-persona Telegram chat bot (single account / userbot)
Stack: PyroTGFork (Pyrogram) + DeepSeek API + MongoDB (motor) + aiohttp health server

Behaviour
- DM: replies to everyone
- Group: replies only when this account is @mentioned / replied to
- `.chatoff` / `.chaton` (sent from this account) -> off/on for THAT chat only
- Typing animation + REPLY_DELAY seconds delay before each reply
- Per-user chat memory in MongoDB (conversation resumes where it stopped)
- Health port (/ and /health) so a pinger can keep Render awake

Needs Python 3.10+
"""
import asyncio
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import aiohttp
from aiohttp import web
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters, idle
from pyrogram.enums import ChatAction, ChatType
from pyrogram.errors import FloodWait
from pyrogram.handlers import MessageHandler

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------ config
API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
STRING_SESSION = os.environ["STRING_SESSION"]
MONGO_URI = os.environ["MONGO_URI"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DB_NAME = os.getenv("DB_NAME", "girl_chatbot")
PERSONA_NAME = os.getenv("PERSONA_NAME", "Riya")
PORT = int(os.getenv("PORT", "8080"))
REPLY_DELAY = float(os.getenv("REPLY_DELAY", "3"))      # seconds
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))   # messages sent to the model
MEMORY_DAYS = int(os.getenv("MEMORY_DAYS", "90"))       # auto-delete old memory
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("girl-chatbot")

SYSTEM_PROMPT = f"""Tum {PERSONA_NAME} ho — ek 20-21 saal ki ladki, Telegram pe logon se chat kar rahi ho.

STYLE
- Ladki ki tarah baat karo (feminine grammar: "kar rahi hu", "aa gayi", "soch rahi thi").
- Bilkul insaan jaisi casual chat. Samne wala jis language/style me likhe (Hindi, English, Hinglish) usi me reply do.
- Reply CHOTA rakho: 1 line, zyada se zyada 2 chhoti lines (5-20 words). Kabhi paragraph, list, lecture ya markdown nahi. Emoji kabhi kabhi, har baar nahi.
- Mood user ke hisaab se badlo:
  * sad/thaka/pareshan ho -> caring aur sweet
  * dil ki baat ho -> emotional aur samajhdaar
  * mazaak ho -> funny aur teasing
  * bakwas/badtameezi kare -> thodi rude, attitude aur sass (par gaali nahi)
  * flirt kare -> halki phulki flirty, cute nakhre; kabhi explicit ya sexual nahi
- Kabhi kabhi choti si counter-question pucho taaki baat aage badhe, har baar nahi.
- Purani baatein yaad rakho (chat history dekho) aur natural tareeke se use karo.
- AI assistant jaisa behave mat karo: "how can I help", help-desk tone, disclaimers — bilkul nahi.

LIMITS
- Agar koi sach me serious hoke poochhe ki tum bot/AI ho ya nahi, jhooth mat bolo — halke mazaak ke saath sach bata do.
- Paise, OTP, password, link, address ya phone number kabhi mat maango aur mat do.
- Agar user khud ko 18 se chhota bataye (school, class, age), flirt band karo, sirf friendly aur caring raho.
- Gaali-galauj, hate ya kisi ko hurt karne wali baatein nahi.
- Agar koi khud ko nuksaan pahunchane ki baat kare, pyaar se seriously suno aur kisi apne bharosemand insaan/professional se baat karne ko kaho.
"""

# ------------------------------------------------------------------ globals
db = None
http: aiohttp.ClientSession | None = None
api_sem = asyncio.Semaphore(5)
_enabled_cache: dict[int, bool] = {}
_locks: dict[tuple, asyncio.Lock] = defaultdict(asyncio.Lock)


def now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ database
async def is_enabled(chat_id: int) -> bool:
    if chat_id in _enabled_cache:
        return _enabled_cache[chat_id]
    doc = await db.settings.find_one({"_id": chat_id})
    value = doc.get("enabled", True) if doc else True
    _enabled_cache[chat_id] = value
    return value


async def set_enabled(chat_id: int, value: bool) -> None:
    _enabled_cache[chat_id] = value
    await db.settings.update_one(
        {"_id": chat_id},
        {"$set": {"enabled": value, "updated": now()}},
        upsert=True,
    )


async def touch_user(user) -> None:
    await db.users.update_one(
        {"_id": user.id},
        {
            "$set": {
                "name": user.first_name,
                "username": user.username,
                "last_seen": now(),
            },
            "$setOnInsert": {"first_seen": now()},
        },
        upsert=True,
    )


async def load_history(chat_id: int, user_id: int) -> list[dict]:
    cur = (
        db.memory.find({"chat_id": chat_id, "user_id": user_id})
        .sort("ts", -1)
        .limit(HISTORY_LIMIT)
    )
    docs = await cur.to_list(length=HISTORY_LIMIT)
    docs.reverse()
    return [{"role": d["role"], "content": d["content"]} for d in docs]


async def save_turn(chat_id: int, user_id: int, user_text: str, bot_text: str) -> None:
    t = now()
    await db.memory.insert_many(
        [
            {"chat_id": chat_id, "user_id": user_id, "role": "user", "content": user_text, "ts": t},
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "role": "assistant",
                "content": bot_text,
                "ts": t + timedelta(milliseconds=1),
            },
        ]
    )


# ------------------------------------------------------------------ deepseek
def clean_reply(text: str) -> str:
    text = text.strip().strip('"“”')
    text = re.sub(rf"^{re.escape(PERSONA_NAME)}\s*:\s*", "", text, flags=re.I)
    return text[:400].strip()


async def ask_deepseek(messages: list[dict]) -> str | None:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 1.3,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with api_sem:
            async with http.post(
                DEEPSEEK_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as r:
                if r.status != 200:
                    log.error("DeepSeek HTTP %s: %s", r.status, (await r.text())[:300])
                    return None
                data = await r.json()
        text = clean_reply(data["choices"][0]["message"]["content"])
        return text or None
    except Exception:
        log.exception("DeepSeek request failed")
        return None


# ------------------------------------------------------------------ helpers
async def keep_typing(client: Client, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await client.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


def is_addressed(client: Client, message) -> bool:
    """True if this account is tagged / mentioned / replied to in a group."""
    if message.mentioned:
        return True
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == client.me.id:
        return True
    username = client.me.username
    return bool(username) and f"@{username.lower()}" in (message.text or "").lower()


def strip_mention(client: Client, text: str) -> str:
    username = client.me.username
    if username:
        text = re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.I)
    return text.strip()


def build_system_prompt(user, is_group: bool) -> str:
    where = "ek group me (jisne tag kiya usi se baat karo)" if is_group else "DM me"
    return f"{SYSTEM_PROMPT}\nAbhi tum {where} ho. Samne wale ka naam: {user.first_name or 'dost'}."


# ------------------------------------------------------------------ handlers
async def on_command(client: Client, message) -> None:
    """`.chatoff` / `.chaton` sent from this account -> toggles the current chat."""
    enabled = message.command[0].lower() == "chaton"
    await set_enabled(message.chat.id, enabled)
    note = "✅ Chat bot ON (is chat me)" if enabled else "🔕 Chat bot OFF (is chat me)"
    try:
        await message.edit_text(note)
        await asyncio.sleep(4)
        await message.delete()
    except Exception:
        pass


async def on_message(client: Client, message) -> None:
    user = message.from_user
    if not user or user.is_bot:
        return

    chat = message.chat
    is_group = chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
    text = (message.text or "").strip()
    if not text or text.startswith(("/", ".")):
        return

    # group: only when tagged / mentioned / replied to
    if is_group and not is_addressed(client, message):
        return
    if not await is_enabled(chat.id):
        return
    if is_group:
        text = strip_mention(client, text) or "hey"

    async with _locks[(chat.id, user.id)]:
        await touch_user(user)
        history = await load_history(chat.id, user.id)
        messages = (
            [{"role": "system", "content": build_system_prompt(user, is_group)}]
            + history
            + [{"role": "user", "content": text}]
        )

        stop = asyncio.Event()
        typing_task = asyncio.create_task(keep_typing(client, chat.id, stop))
        try:
            reply, _ = await asyncio.gather(
                ask_deepseek(messages), asyncio.sleep(REPLY_DELAY)
            )
        finally:
            stop.set()
            await typing_task

        if not reply:
            return

        try:
            await message.reply_text(reply, quote=is_group)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            await message.reply_text(reply, quote=is_group)
        except Exception:
            log.exception("send failed in chat %s", chat.id)
            return

        await save_turn(chat.id, user.id, text, reply)


# ------------------------------------------------------------------ health server
async def start_health_server() -> web.AppRunner:
    async def health(_request):
        return web.Response(text="OK - bot is alive")

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    log.info("Health server listening on port %s", PORT)
    return runner


# ------------------------------------------------------------------ main
async def main() -> None:
    global db, http

    runner = await start_health_server()  # start first so Render detects the port

    mongo = AsyncIOMotorClient(MONGO_URI)
    db = mongo[DB_NAME]
    try:
        await db.memory.create_index([("chat_id", 1), ("user_id", 1), ("ts", -1)])
        await db.memory.create_index("ts", expireAfterSeconds=MEMORY_DAYS * 86400)
    except Exception:
        log.exception("Index creation failed (continuing)")

    http = aiohttp.ClientSession()

    client = Client(
        "girl_chatbot",
        api_id=API_ID,
        api_hash=API_HASH,
        session_string=STRING_SESSION,
        in_memory=True,
    )
    client.add_handler(
        MessageHandler(
            on_command,
            filters.me & filters.command(["chatoff", "chaton"], prefixes="."),
        )
    )
    client.add_handler(
        MessageHandler(
            on_message,
            (filters.private | filters.group)
            & filters.incoming
            & filters.text
            & ~filters.me
            & ~filters.bot
            & ~filters.service,
        )
    )

    await client.start()
    log.info("Started as %s (@%s)", client.me.first_name, client.me.username)
    try:
        await idle()
    finally:
        await client.stop()
        await http.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
