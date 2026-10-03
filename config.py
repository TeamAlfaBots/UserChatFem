"""
config.py - saari settings/values ek jagah.

Zyadatar values environment variables se aati hain (Render ke Environment tab me,
ya local `.env` file me). Yahan defaults, persona prompt aur word lists hain.
Kuch badalna ho to yahi file (ya env vars) badlo, main.py ko chhune ki zaroorat nahi.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------ required (secrets)
_REQUIRED = ("API_ID", "API_HASH", "STRING_SESSION", "MONGO_URI", "DEEPSEEK_API_KEY")
_missing = [n for n in _REQUIRED if not os.getenv(n)]
if _missing:
    raise RuntimeError("Missing required environment variables: " + ", ".join(_missing))

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
STRING_SESSION = os.environ["STRING_SESSION"]
MONGO_URI = os.environ["MONGO_URI"]

# ------------------------------------------------------------------ AI providers
# primary = DEEPSEEK_* (kisi bhi OpenAI-compatible API ke liye), backups = FALLBACK / FALLBACK2 / FALLBACK3
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PROVIDER_CONFIGS = [
    {"name": "primary", "url": DEEPSEEK_URL, "key": DEEPSEEK_API_KEY, "model": DEEPSEEK_MODEL}
]
for _prefix in ("FALLBACK", "FALLBACK2", "FALLBACK3"):
    _url = os.getenv(f"{_prefix}_URL")
    _key = os.getenv(f"{_prefix}_API_KEY")
    _model = os.getenv(f"{_prefix}_MODEL")
    if _url and _key and _model:
        PROVIDER_CONFIGS.append(
            {"name": _prefix.lower(), "url": _url, "key": _key, "model": _model}
        )

# generation settings
MAX_TOKENS = _int("MAX_TOKENS", 120)
TEMPERATURE = _float("TEMPERATURE", 1.1)
TOP_P = _float("TOP_P", 0.95)
FREQUENCY_PENALTY = _float("FREQUENCY_PENALTY", 0.3)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 45)      # seconds per API request
API_CONCURRENCY = _int("API_CONCURRENCY", 5)       # parallel API requests

# provider cooldowns (seconds) after an error
COOLDOWN_DAILY = _int("COOLDOWN_DAILY", 1800)      # daily limit / bad key / no balance / wrong model
COOLDOWN_RATE = _int("COOLDOWN_RATE", 60)          # normal 429 (per-minute limit)
COOLDOWN_ERROR = _int("COOLDOWN_ERROR", 30)        # 5xx / timeouts

# ------------------------------------------------------------------ database / memory
DB_NAME = os.getenv("DB_NAME", "girl_chatbot")
HISTORY_LIMIT = _int("HISTORY_LIMIT", 20)          # messages sent to the model
MEMORY_DAYS = _int("MEMORY_DAYS", 90)              # auto-delete old memory after N days

# ------------------------------------------------------------------ behaviour
PERSONA_NAME = os.getenv("PERSONA_NAME", "Riya")
REPLY_DELAY = _float("REPLY_DELAY", 3)             # seconds (typing animation + delay)
MAX_BUBBLES = _int("MAX_BUBBLES", 3)               # max messages per reply (split by ||)
MAX_REPLY_CHARS = _int("MAX_REPLY_CHARS", 400)
CMD_PREFIX = os.getenv("CMD_PREFIX", ".")          # .chatoff / .chaton / .stickers
PORT = _int("PORT", 8080)                          # health server port (Render sets PORT)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ------------------------------------------------------------------ stickers
STICKER_REPLY_CHANCE = _float("STICKER_REPLY_CHANCE", 0.7)   # 0..1
STICKER_ANY = _bool("STICKER_ANY", False)          # no emoji match -> random sticker (else text)
STICKER_REFRESH_MIN = _int("STICKER_REFRESH_MIN", 30)
STICKER_MAX_SETS = _int("STICKER_MAX_SETS", 40)

# ------------------------------------------------------------------ chat style learning
STYLE_LEARNING = _bool("STYLE_LEARNING", True)
STYLE_SAMPLE = _int("STYLE_SAMPLE", 30)            # recent user msgs analysed
STYLE_MIN_MSGS = _int("STYLE_MIN_MSGS", 5)         # start adapting after N msgs

# word lists used by style learning
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

# ------------------------------------------------------------------ persona prompt
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
