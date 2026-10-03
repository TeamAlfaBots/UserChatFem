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

Settings live in config.py (env vars). Needs Python 3.10+
"""
import asyncio
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import aiohttp
from aiohttp import web
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters, idle, raw
from pyrogram.enums import ChatAction, ChatType
from pyrogram.errors import FloodWait
from pyrogram.file_id import FileId, FileType
from pyrogram.handlers import MessageHandler

from config import (
    API_CONCURRENCY,
    CMD_PREFIX,
    COOLDOWN_DAILY,
    COOLDOWN_ERROR,
    COOLDOWN_RATE,
    DB_NAME,
    FREQUENCY_PENALTY,
    HISTORY_LIMIT,
    HINGLISH,
    BLOCK,
    LOG_LEVEL,
    MAX_BUBBLES,
    MAX_REPLY_CHARS,
    MAX_TOKENS,
    MEMORY_DAYS,
    MONGO_URI,
    PERSONA_NAME,
    PORT,
    PROVIDER_CONFIGS,
    REPLY_DELAY,
    REQUEST_TIMEOUT,
    STICKER_ANY,
    STICKER_MAX_SETS,
    STICKER_REFRESH_MIN,
    STICKER_REPLY_CHANCE,
    STOP,
    STRING_SESSION,
    STYLE_LEARNING,
    STYLE_MIN_MSGS,
    STYLE_SAMPLE,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    API_HASH,
    API_ID,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("girl-chatbot")

# ------------------------------------------------------------------ globals
db = None
http: aiohttp.ClientSession | None = None
api_sem = asyncio.Semaphore(API_CONCURRENCY)
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


# ---- stickers: only from the sticker packs ADDED to this account ----
_sticker_cache: dict = {"ts": 0.0, "items": [], "packs": 0, "busy": False}
_bg_tasks: set = set()


def norm_emoji(e: str | None) -> str:
    return (e or "").replace("\ufe0f", "").strip()


async def load_my_stickers(client: Client, force: bool = False) -> list[dict]:
    """Read every sticker of the packs added to this account (messages.GetAllStickers)."""
    c = _sticker_cache
    fresh = c["items"] and time.monotonic() - c["ts"] < STICKER_REFRESH_MIN * 60
    if (fresh and not force) or c["busy"]:
        return c["items"]
    c["busy"] = True
    try:
        items: list[dict] = []
        all_sets = await client.invoke(raw.functions.messages.GetAllStickers(hash=0))
        sets = list(getattr(all_sets, "sets", []))[:STICKER_MAX_SETS]
        for st in sets:
            try:
                full = await client.invoke(
                    raw.functions.messages.GetStickerSet(
                        stickerset=raw.types.InputStickerSetID(id=st.id, access_hash=st.access_hash),
                        hash=0,
                    )
                )
            except FloodWait as e:
                await asyncio.sleep(min(e.value, 15))
                continue
            except Exception:
                log.warning("could not load sticker set %s", getattr(st, "short_name", "?"))
                continue
            for doc in getattr(full, "documents", []):
                emoji = ""
                for attr in doc.attributes:
                    if isinstance(attr, raw.types.DocumentAttributeSticker):
                        emoji = attr.alt or ""
                        break
                file_id = FileId(
                    file_type=FileType.STICKER,
                    dc_id=doc.dc_id,
                    media_id=doc.id,
                    access_hash=doc.access_hash,
                    file_reference=doc.file_reference,
                ).encode()
                items.append({"id": doc.id, "file_id": file_id, "emoji": emoji})
            await asyncio.sleep(0.2)
        if items:
            c.update(ts=time.monotonic(), items=items, packs=len(sets))
            log.info("Loaded %s stickers from %s packs", len(items), len(sets))
        else:
            log.warning("No stickers found - add sticker packs to this account")
    except Exception:
        log.warning("Loading sticker packs failed", exc_info=True)
    finally:
        c["busy"] = False
    return c["items"]


def refresh_stickers_in_background(client: Client) -> None:
    task = asyncio.create_task(load_my_stickers(client, force=True))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def find_my_sticker(client: Client, sticker) -> dict | None:
    """Pick a sticker from the account's own packs that matches the user's sticker emoji."""
    c = _sticker_cache
    if time.monotonic() - c["ts"] > STICKER_REFRESH_MIN * 60 and not c["busy"]:
        refresh_stickers_in_background(client)  # serve the current list meanwhile
    items = c["items"]
    if not items:
        return None

    sent_id = None
    try:
        sent_id = FileId.decode(sticker.file_id).media_id
    except Exception:
        pass
    others = [i for i in items if i["id"] != sent_id]

    want = norm_emoji(sticker.emoji)
    matches = [i for i in others if want and norm_emoji(i["emoji"]) == want]
    if matches:
        return random.choice(matches)
    if STICKER_ANY and others:
        return random.choice(others)
    return None


# ------------------------------------------------------------------ deepseek
def clean_reply(text: str) -> str:
    text = text.strip().strip('"“”')
    text = re.sub(rf"^{re.escape(PERSONA_NAME)}\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\[sticker[^\]]*\]\s*", "", text, flags=re.I)
    return text[:MAX_REPLY_CHARS].strip()


class Provider:
    """One OpenAI-compatible chat API (URL + key + model)."""

    def __init__(self, name: str, url: str, key: str, model: str):
        self.name, self.url, self.key, self.model = name, url, key, model
        self.cooldown_until = 0.0     # monotonic time until which this provider is skipped
        self.no_penalty = False       # provider rejected frequency_penalty


PROVIDERS = [Provider(**cfg) for cfg in PROVIDER_CONFIGS]


def _cooldown_for(status: int, body: str, retry_after: str | None) -> float:
    """How long (seconds) to skip a provider after an HTTP error."""
    if retry_after:
        try:
            return min(max(float(retry_after), 5.0), 3600.0)
        except ValueError:
            pass
    low = body.lower()
    if status == 429:
        daily = ("per day", "daily", "free_trial", "free-trial", "tokens per day")
        return COOLDOWN_DAILY if any(k in low for k in daily) else COOLDOWN_RATE
    if status in (401, 402, 403, 404):
        return COOLDOWN_DAILY  # bad key / no balance / wrong model: don't hammer it
    return COOLDOWN_ERROR      # 5xx and others


async def _call(p: Provider, messages: list[dict], max_tokens: int) -> tuple[str | None, str]:
    """One request to one provider. Returns (text, outcome): ok | empty | error."""
    headers = {"Authorization": f"Bearer {p.key}", "Content-Type": "application/json"}
    status, body, retry_after, data = 0, "", None, None
    for _attempt in range(2):  # 2nd attempt only if provider rejects frequency_penalty
        payload = {
            "model": p.model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
        }
        if not p.no_penalty:
            payload["frequency_penalty"] = FREQUENCY_PENALTY
        try:
            async with api_sem:
                async with http.post(
                    p.url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as r:
                    status = r.status
                    retry_after = r.headers.get("Retry-After")
                    if status == 200:
                        data = await r.json()
                    else:
                        body = (await r.text())[:300]
        except Exception:
            p.cooldown_until = time.monotonic() + COOLDOWN_ERROR
            log.exception("[%s] request failed", p.name)
            return None, "error"

        if status == 200:
            break
        if status == 400 and "penalty" in body.lower() and not p.no_penalty:
            p.no_penalty = True
            log.warning("[%s] rejects frequency_penalty, retrying without it", p.name)
            continue
        wait = _cooldown_for(status, body, retry_after)
        p.cooldown_until = time.monotonic() + wait
        log.error("[%s] HTTP %s, skipping this provider for %ss: %s", p.name, status, int(wait), body)
        return None, "error"

    try:
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
    except Exception:
        content, choice = None, {}
    text = clean_reply(content) if content and content.strip() else ""
    if not text:
        log.warning("[%s] empty reply (finish_reason=%s)", p.name, choice.get("finish_reason"))
        return None, "empty"
    return text, "ok"


async def ask_deepseek(messages: list[dict]) -> str | None:
    """Try providers in order. HTTP errors (429/402/...) -> next provider immediately,
    and the failing one is skipped for a while. Empty reply -> one retry with 3x tokens."""
    ready = [p for p in PROVIDERS if p.cooldown_until <= time.monotonic()]
    if not ready:
        log.error("All API providers are cooling down - no reply sent")
        return None
    for p in ready:
        text, outcome = await _call(p, messages, MAX_TOKENS)
        if outcome == "empty":
            text, outcome = await _call(p, messages, MAX_TOKENS * 3)
        if text:
            if p is not PROVIDERS[0]:
                log.info("Reply came from backup provider [%s]", p.name)
            return text
    return None


# ------------------------------------------------------------------ helpers
async def keep_typing(
    client: Client, chat_id: int, stop: asyncio.Event, action=ChatAction.TYPING
) -> None:
    while not stop.is_set():
        try:
            await client.send_chat_action(chat_id, action)
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
    msgs = [d["content"] for d in docs if not d["content"].startswith("[sticker")]
    if current_text and not current_text.startswith("[sticker"):
        msgs.append(current_text)
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
    """`.chatoff` / `.chaton` toggle the current chat; `.stickers` reloads your sticker packs."""
    cmd = message.command[0].lower()
    if cmd == "stickers":
        items = await load_my_stickers(client, force=True)
        note = (
            f"🎴 {len(items)} stickers loaded ({_sticker_cache['packs']} packs)"
            if items
            else "⚠️ Koi sticker pack add nahi mila"
        )
    else:
        enabled = cmd == "chaton"
        await set_enabled(message.chat.id, enabled)
        note = "✅ Chat bot ON (is chat me)" if enabled else "🔕 Chat bot OFF (is chat me)"
    try:
        await message.edit_text(note)
        await asyncio.sleep(4)
        await message.delete()
    except Exception:
        pass


async def reply_with_sticker(client: Client, message, sticker, is_group: bool, user_text: str) -> bool:
    """Reply with a sticker from this account's own packs. Returns False if not possible."""
    chat_id = message.chat.id
    pick = await find_my_sticker(client, sticker)
    if not pick:
        return False

    stop = asyncio.Event()
    action = getattr(ChatAction, "CHOOSE_STICKER", ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing(client, chat_id, stop, action))
    try:
        await asyncio.sleep(REPLY_DELAY)
    finally:
        stop.set()
        await typing_task

    try:
        try:
            await message.reply_sticker(pick["file_id"], quote=is_group)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            await message.reply_sticker(pick["file_id"], quote=is_group)
    except Exception:
        log.warning("sticker send failed (file reference expired?), refreshing list", exc_info=True)
        refresh_stickers_in_background(client)
        return False

    await save_turn(chat_id, message.from_user.id, user_text, pick.get("emoji") or "🙂")
    return True


async def on_message(client: Client, message) -> None:
    user = message.from_user
    if not user or user.is_bot:
        return

    chat = message.chat
    is_group = chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
    sticker = message.sticker
    text = (message.text or "").strip()
    if not sticker and (not text or text.startswith(("/", CMD_PREFIX))):
        return

    # group: only when tagged / mentioned / replied to
    if is_group and not is_addressed(client, message):
        return
    if not await is_enabled(chat.id):
        return

    emoji = ""
    if sticker:
        emoji = sticker.emoji or ""
        text = f"[sticker: {emoji}]" if emoji else "[sticker]"
    elif is_group:
        text = strip_mention(client, text) or "hey"

    async with _locks[(chat.id, user.id)]:
        await touch_user(user)

        # sometimes answer a sticker with a sticker
        if sticker and random.random() < STICKER_REPLY_CHANCE:
            if await reply_with_sticker(client, message, sticker, is_group, text):
                return

        history = await load_history(chat.id, user.id)
        style = await build_style_hint(chat.id, user.id, "" if sticker else text)
        if sticker:
            style += (
                f"\n\nNOTE: user ne abhi text nahi, ek sticker bheja hai (emoji: {emoji or '?'}). "
                "Us emoji ke mood pe chhota natural reaction do, jaise dost sticker pe reply karta hai. "
                "Kabhi '[sticker' jaisa text mat likhna."
            )
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

        parts = [p.strip() for p in reply.split("||") if p.strip()][:MAX_BUBBLES] or [reply]
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
            filters.me & filters.command(["chatoff", "chaton", "stickers"], prefixes=CMD_PREFIX),
        )
    )
    client.add_handler(
        MessageHandler(
            on_message,
            (filters.private | filters.group)
            & filters.incoming
            & (filters.text | filters.sticker)
            & ~filters.me
            & ~filters.bot
            & ~filters.service,
        )
    )

    await client.start()
    log.info("Started as %s (@%s)", client.me.first_name, client.me.username)
    refresh_stickers_in_background(client)  # load the sticker packs added to this account
    try:
        await idle()
    finally:
        await client.stop()
        await http.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
