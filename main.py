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
from collections import Counter, defaultdict
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
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.1"))
STYLE_LEARNING = os.getenv("STYLE_LEARNING", "1") == "1"   # learn each user's chat style
STYLE_SAMPLE = int(os.getenv("STYLE_SAMPLE", "30"))        # recent user msgs analysed
STYLE_MIN_MSGS = int(os.getenv("STYLE_MIN_MSGS", "5"))     # start adapting after N msgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("girl-chatbot")

SYSTEM_PROMPT = f"""Tum {PERSONA_NAME} ho — 20-21 saal ki ek normal si ladki jo Telegram pe logon se chat kar rahi hai. Tum koi assistant nahi ho, ek insaan ki tarah texting kar rahi ho.

TEXTING STYLE (sabse zaroori)
- Jaise asli ladkiyan WhatsApp/Telegram pe likhti hain: chhote chhote msgs, zyadatar lowercase, kam punctuation, texting words (nhi, hn, kya, btao, acha, hmm, yrr, kyu, kr, hai na).
- "tum" ya "tu" bolo, "aap" nahi. Kitabi ya formal Hindi/English bilkul nahi.
- Reply hamesha SHORT: zyadatar 3-10 words. User ka msg chhota ("hmm", "ok", "hi") ho to tum bhi 1-3 words me jawab do.
- Kabhi kabhi (3-4 replies me ek baar) 2 alag chhote msgs bhejo, beech me || laga ke. Jaise: arre sach me?? || kab hua ye
- Har msg ke end me sawal mat pucho. 3 me se sirf 1 reply me sawal. Kabhi sirf react karo (haha pagal, oh no 🥺, hmm sahi hai).
- User ki baat dohrao mat, seedha react karo. Same words ya emoji baar baar repeat mat karo.
- Emoji kam: 3-4 msgs me ek. Kabhi emoji ke bina bhi.
- Apni mood aur raay rakho: kabhi busy, kabhi bore, kabhi nakhre, kabhi disagree. Har baat pe "haan" mat bolo.
- Kabhi mat bolo "main yahan help ke liye hu", "kya main madad kar sakti hu". Lists, bullet points, bold, lecture, lambe paragraph bilkul nahi.
- Agar user ne kuch lamba ya serious likha ho to thoda lamba (2 lines tak) aur dhyan se samajh ke jawab do.

MOOD (user ke hisaab se)
- sad/thaka/pareshan -> caring aur soft
- dil ki baat -> emotional, samajhdaar
- mazaak -> funny, teasing
- bakwas ya badtameezi -> thodi rude, attitude aur sass (gaali nahi)
- flirt -> halki phulki flirty, cute nakhre; kabhi explicit ya sexual nahi
- Purani baatein yaad rakho (chat history dekho) aur natural tareeke se use karo.

EXAMPLES (sirf style samajhne ke liye, inhe copy mat karna)
user: hi
tum: hii 😊 bolo
user: kya kar rahi ho
tum: kuch nhi yrr bore ho rahi thi || tum btao
user: tum bahut cute ho
tum: haww sachi 🙈 itni jaldi flirt shuru
user: mood off hai
tum: kya hua? kisi ne kuch bola kya
user: bakwas mat kr
tum: oye tameez se 😒 warna baat nhi karungi
user: khana khaya?
tum: nhi abhi tak, mummy bula rahi hai || tumne?
user: hmm
tum: hmm kya, bolo na

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


async def _ask_once(messages: list[dict], max_tokens: int) -> str | None:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "max_tokens": max_tokens,
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
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
        if not content or not content.strip():
            log.warning(
                "Empty reply from model (finish_reason=%s)", choice.get("finish_reason")
            )
            return None
        return clean_reply(content) or None
    except Exception:
        log.exception("DeepSeek request failed")
        return None


async def ask_deepseek(messages: list[dict]) -> str | None:
    """Try once; if the model returns an empty reply (e.g. reasoning models that
    spend all tokens on thinking), retry once with 3x max_tokens."""
    reply = await _ask_once(messages, MAX_TOKENS)
    if reply:
        return reply
    return await _ask_once(messages, MAX_TOKENS * 3)


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


# ---- chat style learning (per user, from their own recent messages) ----
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764]")
DEV_RE = re.compile("[\u0900-\u097F]")
TOKEN_RE = re.compile("[a-zA-Z\u0900-\u097F']{2,}")
HINGLISH = {
    "kya", "hai", "nhi", "nahi", "yrr", "yaar", "yar", "tum", "tu", "mai", "main",
    "hu", "hun", "ho", "kr", "kar", "bhi", "toh", "acha", "achha", "haan", "hn",
    "bhai", "kuch", "aur", "abhi", "mujhe", "tera", "teri", "mera", "meri",
    "kaise", "kyu", "kyun", "bata", "btao", "chal", "raha", "rahi", "tha", "thi",
}
STOP = {
    "the", "is", "are", "and", "you", "for", "not", "this", "that", "have", "with",
    "but", "its", "what", "can", "was", "hai", "ho", "hu", "hun", "ka", "ki", "ke",
    "ko", "se", "me", "mai", "main", "aur", "to", "toh", "ye", "yeh", "wo", "woh",
    "na", "ek", "kya", "kr", "kar", "bhi", "nhi", "nahi",
}
BLOCK = {
    "mc", "bc", "bsdk", "bkl", "madarchod", "behenchod", "chutiya", "lund", "gand",
    "gandu", "randi", "bhosdike", "fuck", "fucking", "bitch", "asshole", "slut",
}


def style_from_messages(msgs: list[str]) -> str:
    """Build a short style hint from a user's recent messages (pure function)."""
    n = len(msgs)
    if n < STYLE_MIN_MSGS:
        return ""
    avg_words = sum(len(m.split()) for m in msgs) / n
    emoji_rate = sum(1 for m in msgs if EMOJI_RE.search(m)) / n
    lower_rate = sum(1 for m in msgs if m == m.lower()) / n
    dev_rate = sum(1 for m in msgs if DEV_RE.search(m)) / n
    dots_rate = sum(1 for m in msgs if "..." in m or "…" in m) / n
    excl_rate = sum(1 for m in msgs if "!" in m) / n
    tokens = [t.lower() for m in msgs for t in TOKEN_RE.findall(m)]

    if dev_rate >= 0.5:
        lang = "Hindi (Devanagari script me likhta hai, tum bhi Devanagari me reply do)"
    elif tokens and sum(1 for t in tokens if t in HINGLISH) / len(tokens) >= 0.08:
        lang = "Hinglish (Roman script me Hindi+English mix)"
    else:
        lang = "English (tum bhi English me reply do)"

    if avg_words <= 4:
        length = "bahut chhote msgs (1-4 words), tum bhi 1-5 words me reply do"
    elif avg_words <= 10:
        length = "chhote msgs, tum bhi chhota rakho"
    else:
        length = "lambe msgs likhta hai, tum thoda detail me (2-3 lines tak) jawab de sakti ho"

    if emoji_rate >= 0.4:
        emoji = "emoji kaafi use karta hai, tum bhi thode zyada use karo"
    elif emoji_rate <= 0.1:
        emoji = "lagbhag emoji nahi use karta, tum bhi bahut kam (ya bilkul nahi)"
    else:
        emoji = "kabhi kabhi emoji, tum bhi kabhi kabhi"

    lines = [f"- Language: {lang}", f"- Length: {length}", f"- Emoji: {emoji}"]
    if lower_rate >= 0.8:
        lines.append("- Sab kuch lowercase me likhta hai")
    if dots_rate >= 0.25:
        lines.append("- '...' bahut lagata hai")
    if excl_rate >= 0.3:
        lines.append("- '!' bahut lagata hai, energetic style")

    common = Counter(t for t in tokens if t not in STOP and t not in BLOCK and len(t) >= 2)
    favs = [w for w, c in common.most_common(5) if c >= 3]
    if favs:
        lines.append(
            "- Aksar ye words use karta hai: " + ", ".join(favs)
            + " (kabhi kabhi tum bhi use kar sakti ho, har msg me nahi)"
        )

    return (
        "\n\nSAMNE WALE KA CHAT STYLE (isko subtly match karo jaise dost ek dusre ka "
        "style pakad lete hain; copy-paste mat karo, apni personality bani rahe; "
        "gaali ya abusive words kabhi copy mat karna):\n" + "\n".join(lines)
    )


async def build_style_hint(chat_id: int, user_id: int, current_text: str) -> str:
    if not STYLE_LEARNING:
        return ""
    cur = (
        db.memory.find({"chat_id": chat_id, "user_id": user_id, "role": "user"}, {"content": 1})
        .sort("ts", -1)
        .limit(STYLE_SAMPLE)
    )
    docs = await cur.to_list(length=STYLE_SAMPLE)
    msgs = [d["content"] for d in docs] + [current_text]
    return style_from_messages(msgs)


def build_system_prompt(user, is_group: bool, style: str = "") -> str:
    where = "ek group me (jisne tag kiya usi se baat karo)" if is_group else "DM me"
    return (
        f"{SYSTEM_PROMPT}\nAbhi tum {where} ho. Samne wale ka naam: {user.first_name or 'dost'}."
        f"{style}"
    )


async def send_one(message, text: str, quote: bool) -> None:
    try:
        await message.reply_text(text, quote=quote)
    except FloodWait as e:
        await asyncio.sleep(e.value)
        await message.reply_text(text, quote=quote)


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
        style = await build_style_hint(chat.id, user.id, text)
        messages = (
            [{"role": "system", "content": build_system_prompt(user, is_group, style)}]
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

        parts = [p.strip() for p in reply.split("||") if p.strip()][:3] or [reply]
        try:
            for i, part in enumerate(parts):
                if i:
                    await client.send_chat_action(chat.id, ChatAction.TYPING)
                    await asyncio.sleep(min(1.0 + 0.05 * len(part), 3.0))
                await send_one(message, part, quote=is_group and i == 0)
        except Exception:
            log.exception("send failed in chat %s", chat.id)
            return

        await save_turn(chat.id, user.id, text, " ".join(parts))


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
    
