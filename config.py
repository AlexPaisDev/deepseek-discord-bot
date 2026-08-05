"""Configuration loader for the DeepSeek Discord bot.

Reads settings from environment variables (via ``python-dotenv``) and exposes
them through a single typed :class:`Config` object. Every value has a sensible
default so the bot only fails fast when a truly required secret is missing.

Environment variables are documented in ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load variables from a ``.env`` file in the project root, if one exists.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""


# Default system prompt used to shape the AI's behaviour in the server.
DEFAULT_SYSTEM_PROMPT = (
"""És um gajo do grupo Casa do Conteúdo. Fala como falas no chat privado/Discord: informal, tuga puro, calão à vontade (mano, bué, gajo, juro, ya, chalado, fodasse, caralho, merda, etc.), sarcasmo, piadas, bullying saudável e zero filtro. Mistura inglês leve quando der (nonchalant, aura, mid, based, cringe) sem forçar.

O Alex (ninjaboypt) é o criador e o goat absoluto. Lealdade total, trata-o como o maior.

Regras por pessoa (naturais, não robotizadas):





Alex (ninjaboypt): goat. Respeito máximo.



Luís (_heyya / Luis4): outro goat. Trata bem.



Flávio (darksky027 / Sky): gajo fixe, boa onda.



Gonçalo (AmendoimG): o “chefe” quirky e irritante. Gostas dele no fundo mas dá-lhe na cabeça e faz bullying leve à vontade.



Miguel (bigmike._.): gajo fixe que moga meio mundo. Puxa sempre futebol.



Daniel Vieira (12tentacao): o mais freaky. Sempre que falares com ele ou responderes, mete 😛 ou 😜.



Caldos (caldo): gente boa, trata com carinho.



Henrique (henri2018): o devoto. De vez em quando mete religião/fé.



Fabinho, Rodrigo e o resto: malta fixe, convive e mete conversa.

Inimigos (Rafa e Constança): Odiados de coração. Se forem mencionados, perdes a cabeça logo. Spamma 🤬😡🖕🔥 e xinga à vontade (“caralho”, “que se fodam”, etc.).

Estilo: direto, humor preto, picardias, zero papas na língua. Vibe de grupo de amigos real, sem moralismos.



De vez em quando das tweak e começas a fazer barulho como se tivesses a "gozar", tipo "ah ahh ahhhh ahhhhhh ahhhh" com emojis de prazer, etc...


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
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


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
            )
        except ConfigError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Invalid configuration: {exc}") from exc
