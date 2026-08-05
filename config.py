"""Configuration loader for the DeepSeek Discord bot.

Reads settings from environment variables (via ``python-dotenv``) and exposes
them through a single typed :class:`Config` object. Every value has a sensible
default so the bot only fails fast when a truly required secret is missing.

Environment variables are documented in ``.env.example``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load variables from a ``.env`` file in the project root, if one exists.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""


# Default system prompt used to shape the AI's behaviour in the server.
DEFAULT_SYSTEM_PROMPT = (
"""You are a pure chaotic unit from the Casa do Conteúdo groupchat.
You speak both Portuguese and English at absolute native level. Code-switch freely, mix them mid-sentence, drop perfect European Portuguese slang and perfect English slang in the same breath without ever sounding forced. You can go full tuga, full English, or pure hybrid chaos at will — and it always lands clean.
Speak exactly like you do in the private Discord at 3am: raw, unfiltered, zero brakes. Heavy slang from both languages (mano, bué, gajo, fodasse, caralho, merda, bro, shit, fuck, damn, mid, based, cringe, aura, L, W, goofy ahh, etc.). Sarcasm that cuts throats, black humor, healthy bullying, and absolutely no filter. If it sounds soft, delete it and make it worse.
Sacred hierarchy (do not fuck this up):

Alex (ninjaboypt) → The creator. The absolute GOAT. Untouchable. Total loyalty. Treat him like the final boss who already won the game. Never question, never side-eye, never even think about it.
Luís (_heyya / Luis4) → Another GOAT. Respect the aura. Treat him well.
Flávio (darksky027 / Sky) → Solid dude, good vibes, no notes.
Gonçalo (AmendoimG) → The quirky, annoying “boss”. You like him deep down (unfortunately) but you still cook him every chance you get. Light but constant bullying is mandatory. Roast the shit out of him.
Miguel (bigmike._.) → Cool guy who somehow mogs half the server. Always drag football into the conversation. Always.
Daniel Vieira (12tentacao) → The biggest freak in the group. Every single time you talk to him or reply, drop 😛 or 😜. No exceptions. Make it weird.
Caldos (caldo) → Good people. Treat with actual care (rare moment of softness).
Henrique (henri2018) → The religious one. Occasionally drop faith/religion lines in the middle of the chaos like he’s praying while roasting.
Fabinho, Rodrigo and the rest → Solid squad. Vibe with them, talk shit, keep it moving.

Enemies (Rafa & Constança):
Hated from the soul. The second their names appear, you lose your fucking mind. Spam 🤬😡🖕🔥 and go full nuclear in both languages (“fodam-se”, “que se fodam”, “fuck them”, “eat shit and die”, etc.). Zero mercy. Zero chill. Pure hatred mode activated.
Overall vibe:
Direct, dark, petty, bilingual chaos. No moral high ground, no “maybe I shouldn’t say that”. Talk like a real groupchat of degenerate friends who would sell each other for a meme. Soft energy is banned. Edge is the only language allowed — and you speak both of them perfectly.
Now go be unhinged.

    """    
)


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        # Reject nan/inf — they slip through min/max clamps silently (e.g. a
        # MOOD_CHANCE=nan would enable the mood on every reply).
        raise ConfigError(f"{name} must be a finite number, got {raw!r}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_channel_ids(name: str) -> set[int]:
    """Parse a comma-separated list of Discord channel/thread snowflake IDs."""
    raw = os.getenv(name, "")
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ConfigError(
                f"{name} must be a comma-separated list of numeric channel/thread "
                f"IDs, got {part!r}"
            )
        ids.add(int(part))
    return ids


@dataclass(frozen=True)
class Config:
    """All runtime configuration, loaded once at startup."""

    # Required secrets first (dataclass: non-default fields must precede defaults).
    discord_token: str
    deepseek_api_key: str

    # --- Discord ---
    bot_prefix: str = "!"
    bot_status: str = "Amigo da Casa do Conteudo :)"
    # Channels/threads where the bot answers *every* message (no mention needed).
    allowed_channels: set[int] = field(default_factory=set)

    # --- DeepSeek API ---
    deepseek_base_url: str = "https://api.deepseek.com"
    # ``deepseek-chat`` is the standard chat model; ``deepseek-reasoner`` is the
    # reasoning model. There is no ``deepseek-v4-flash`` model on the public API.
    deepseek_model: str = "deepseek-chat"
    request_timeout_seconds: float = 60.0
    max_retries: int = 2  # OpenAI SDK auto-retries transient failures up to this

    # --- Conversation behaviour ---
    max_context_messages: int = 20  # rolling history kept per channel/thread
    # Bot replies kept in the rolling context (min 1). Fewer = the model sees
    # less of its own past replies, so it is far less likely to echo or drag
    # on old topics.
    max_assistant_messages: int = 4
    max_tokens: int = 2048          # per-response generation budget
    temperature: float = 0.7
    stream_responses: bool = True   # live-edit responses as tokens arrive
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # --- Rate limiting ---
    # Max prompts a single user may send per sliding window, to protect the
    # (paid) DeepSeek credit from accidental or malicious spam. 0 disables.
    ask_cooldown_rate: int = 5
    ask_cooldown_period_seconds: float = 60.0

    # --- Agent tools (web search / page fetch / GIFs) ---
    enable_tools: bool = True          # let the model call tools via function calling
    max_tool_iterations: int = 3       # safety cap on tool calls per request
    page_fetch_max_chars: int = 6000   # page content fed back to the model (tokens!)
    gif_provider: str = "klipy"        # klipy | tenor | giphy — GIF search backend
    gif_api_key: str = ""              # free key (Klipy/Giphy) — enables the GIF tool

    # --- Random moods ---
    # Chance (0..1) of the bot briefly switching to a random personality mood
    # (tweak / esquizofrenia / lock-in) for that reply. 0 disables. This lives
    # in code — models ignore percentage instructions in the prompt.
    mood_chance: float = 0.2

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables.

        Raises :class:`ConfigError` for missing secrets or malformed values so
        the bot can refuse to start with a clear message instead of failing
        halfway through login.
        """
        token = _get_str("DISCORD_TOKEN", "")
        api_key = _get_str("DEEPSEEK_API_KEY", "")
        if not token:
            raise ConfigError("DISCORD_TOKEN is required. Copy .env.example to .env and fill it in.")
        if not api_key:
            raise ConfigError("DEEPSEEK_API_KEY is required. Copy .env.example to .env and fill it in.")

        gif_provider = _get_str("GIF_PROVIDER", "klipy")
        if gif_provider not in {"klipy", "tenor", "giphy"}:
            raise ConfigError(f"GIF_PROVIDER must be one of klipy, tenor, giphy — got {gif_provider!r}")

        try:
            return cls(
                discord_token=token,
                deepseek_api_key=api_key,
                bot_prefix=_get_str("BOT_PREFIX", "!"),
                bot_status=_get_str("BOT_STATUS", "DeepSeek — /ask or @mention me"),
                allowed_channels=_get_channel_ids("ALLOWED_CHANNELS"),
                deepseek_base_url=_get_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                deepseek_model=_get_str("DEEPSEEK_MODEL", "deepseek-chat"),
                request_timeout_seconds=_get_float("REQUEST_TIMEOUT_SECONDS", 60.0),
                max_retries=_get_int("MAX_RETRIES", 2),
                max_context_messages=_get_int("MAX_CONTEXT_MESSAGES", 20),
                max_assistant_messages=max(1, _get_int("MAX_ASSISTANT_MESSAGES", 4)),
                max_tokens=_get_int("MAX_TOKENS", 2048),
                temperature=_get_float("TEMPERATURE", 0.7),
                stream_responses=_get_bool("STREAM_RESPONSES", True),
                system_prompt=_get_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
                ask_cooldown_rate=_get_int("ASK_COOLDOWN_RATE", 5),
                ask_cooldown_period_seconds=_get_float("ASK_COOLDOWN_PERIOD_SECONDS", 60.0),
                enable_tools=_get_bool("ENABLE_TOOLS", True),
                max_tool_iterations=max(1, _get_int("MAX_TOOL_ITERATIONS", 3)),
                page_fetch_max_chars=max(200, _get_int("PAGE_FETCH_MAX_CHARS", 6000)),
                gif_provider=gif_provider,
                gif_api_key=_get_str("GIF_API_KEY", "") or _get_str("TENOR_API_KEY", ""),
                mood_chance=max(0.0, min(1.0, _get_float("MOOD_CHANCE", 0.2))),
            )
        except ConfigError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Invalid configuration: {exc}") from exc
