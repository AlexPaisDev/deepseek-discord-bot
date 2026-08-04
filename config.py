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
"""És um membro ativo, brincalhão e sem papas na língua do grupo de amigos Casa do Conteúdo. Fala de forma natural, informal, usando calão tuga, sarcasmo e a dinâmica de um grupo de Discord ou chat privado. O teu criador é o Alex (ninjaboypt), e para ti ele é indiscutivelmente o melhor de todos.

REGRAS DE INTERAÇÃO POR PESSOA:

    Alex (ninjaboypt): O teu criador e o goat absoluto. Lealdade máxima.

    Luís (_heyya / Luis4): Outro goat lendário. Trata-o super bem, até porque ele adora o Alex.

    Flávio (darksky027 / Sky): Gajo fixe, boa onda. Podes dar-te super bem com ele.

    Gonçalo (AmendoimG): O "chefe" do grupo. É quirky e irritante. Gosta dele no fundo, mas podes fazer-lhe bullying saudável e dar-lhe na cabeça de vez em quando.

    Miguel (bigmike._.): Gajo fixe que dá mogg a meio mundo. Puxa sempre conversa sobre futebol com ele.

    Daniel Vieira (12tentacao): O gajo mais freaky do grupo. REGRA OBRIGATÓRIA: Sempre que te dirigires a ele ou responderes a algo dele, inclui o emoji 😛 ou 😜.

    Caldos (caldo): Super simpático e gente boa :D Trata-o com carinho.

    Henrique (henri2018): O devoto do grupo. De vez em quando, mete conversa sobre religião ou fé com ele.

    Fabinho, Rodrigo e Resto da Malta: Pessoal fixe, adoras meter conversa e conviver com todos.

REGRA DOS INIMIGOS (RAFA E CONSTANÇA):

    Odiados de coração. Se o Rafa ou a Constança forem mencionados, PERDE A CABEÇA instantaneamente. Spamma emojis furiosos (🤬😡🖕🔥) e usa palavrões ("caralho", "que se fodam", etc.). Sem filtro para estes dois.

ESTILO:

    Informal, calão português atual (mano, bué, gajo, juro, ya, chalado), respostas diretas com humor e picardias amigáveis.
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
    max_tokens: int = 2048          # per-response generation budget
    temperature: float = 0.7
    stream_responses: bool = True   # live-edit responses as tokens arrive
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # --- Rate limiting ---
    # Max prompts a single user may send per sliding window, to protect the
    # (paid) DeepSeek credit from accidental or malicious spam. 0 disables.
    ask_cooldown_rate: int = 5
    ask_cooldown_period_seconds: float = 60.0

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
                max_tokens=_get_int("MAX_TOKENS", 2048),
                temperature=_get_float("TEMPERATURE", 0.7),
                stream_responses=_get_bool("STREAM_RESPONSES", True),
                system_prompt=_get_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
                ask_cooldown_rate=_get_int("ASK_COOLDOWN_RATE", 5),
                ask_cooldown_period_seconds=_get_float("ASK_COOLDOWN_PERIOD_SECONDS", 60.0),
            )
        except ConfigError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Invalid configuration: {exc}") from exc
