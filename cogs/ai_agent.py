"""AI Agent cog — all DeepSeek API interaction lives here.

Responsibilities:
  * Slash + prefix commands (``ask``, ``clear``, ``context``, ``ping``, ``help``)
    built with :func:`discord.ext.commands.hybrid_command` so one definition
    powers both interfaces.
  * An ``on_message`` listener that auto-replies when the bot is @mentioned,
    messaged in DMs, or spoken to in a designated channel/thread.
  * Per-channel/thread rolling conversation memory (a bounded deque) so context
    stays fresh without burning tokens.
  * A per-user rate limiter so nobody can drain the paid DeepSeek credit by
    spamming prompts.
  * Per-channel serialisation (asyncio.Lock) so concurrent prompts never
    interleave inside a conversation's memory.
  * Streaming responses with live message edits (throttled to respect Discord
    rate limits), falling back to a typing indicator when streaming is off.
  * Friendly error embeds for rate limits, timeouts, invalid keys, permissions.
  * Splitting of long replies into Discord-safe <=1900-char chunks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from contextlib import nullcontext

import discord
from discord.ext import commands
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from config import Config

import tools as agent_tools

log = logging.getLogger(__name__)

# Discord hard limit is 2000 chars per message; keep a safety margin.
SAFE_MESSAGE_LIMIT = 1900
# Live-edit throttling: stay comfortably under Discord's ~5 edits / 5s per channel.
EDIT_THROTTLE_SECONDS = 0.9
EDIT_MIN_DELTA_CHARS = 100

# OpenAI-compatible APIs accept an optional ``name`` on user messages to
# identify the speaker; it must be a 1-64 char identifier of letters, digits,
# underscores or hyphens. Anything else (spaces, dots, unicode, …) falls back
# to embedding the name in the content instead.
API_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Appended to whatever system prompt is configured, so the anti-repetition
# rules always apply without touching the user's persona.
CONVERSATION_GUIDANCE = (
    "REGRAS DE CONVERSA (aplicam-se sempre):\n"
    "- Responde APENAS à última mensagem do utilizador; as anteriores são só contexto de fundo.\n"
    "- NUNCA te repitas nem ecoes respostas que já deste antes.\n"
    "- Se a conversa mudou de tema, larga o tema antigo imediatamente e não o arrastes.\n"
    "- Respostas diretas; não te alongues nem repitas o que já foi dito."
)

# Appended to the system prompt only when tools are enabled.
TOOL_GUIDANCE = (
    "FERRAMENTAS DISPONÍVEIS (usa quando fizer sentido):\n"
    "- web_search: pesquisa na web (factos atuais, notícias, preços, algo que não saibas de cor).\n"
    "- fetch_page: lê o conteúdo de um URL (ex.: de uma pesquisa) para obteres detalhes.\n"
    "- search_gifs: procura um GIF na Tenor (humor, reações, celebrações).\n"
    "- Só chama uma ferramenta se realmente ajudar; para conversa normal, não chames nenhuma.\n"
    "- Usa as ferramentas EM SILÊNCIO: nunca digas ao utilizador o que estás a fazer, nem mostres URLs ou resultados crus — responde diretamente com a informação pedida (escreve a letra, o resumo, etc.).\n"
    "- Se uma pesquisa ou leitura falhar ou vier truncada, tenta outra fonte antes de desistir.\n"
    "- Se o conteúdo devolvido tiver a marca …[truncado], avisa que está incompleto e dá a parte que tens."
)

TOOL_STATUS_EMOJI = {"web_search": "🔎", "fetch_page": "📄", "search_gifs": "🖼️"}


def _api_user_message(content: str, name: str) -> dict[str, str]:
    """Build a user message tagged with the sender's username.

    Uses the API's native ``name`` field when the username is a valid
    identifier; otherwise prefixes the content with it.
    """
    if API_NAME_RE.fullmatch(name):
        return {"role": "user", "name": name, "content": content}
    return {"role": "user", "content": f"{name}: {content}"}


def chunk_text(text: str, limit: int = SAFE_MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Prefers to break on line boundaries; over-long single lines are hard-split.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        # A single line longer than the limit must be hard-split.
        if len(line) > limit:
            if current:
                chunks.append(current.rstrip())
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit].rstrip())
            continue
        if len(current) + len(line) > limit:
            chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current:
        chunks.append(current.rstrip())
    return chunks


def _strip_mentions(text: str, bot_id: int) -> str:
    """Remove the bot's own mentions (normal and nickname forms) from a prompt."""
    text = text.replace(f"<@{bot_id}>", " ").replace(f"<@!{bot_id}>", " ")
    return text


class ConversationMemory:
    """Rolling conversation context for a single channel or thread.

    Uses a ``deque(maxlen=...)`` so the oldest messages are dropped
    automatically once the budget is exceeded — efficient and predictable.
    User messages carry the sender's username (via the API ``name`` field, or
    as a content prefix when the username isn't a valid identifier) so the
    model knows who is speaking.
    """

    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self.messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add_user(self, content: str, name: str | None = None) -> None:
        self.messages.append(_api_user_message(content, name) if name else {"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def to_api_messages(self, system_prompt: str, max_assistant: int | None = None) -> list[dict[str, str]]:
        """Messages ready to send to the API (system prompt + trimmed history).

        When ``max_assistant`` is set, keeps only the tail of the conversation:
        the last ``max_assistant`` bot replies — plus the user message that
        prompted the oldest kept reply — and everything after it. Cutting the
        bot's own older replies stops the model from echoing its past monologue
        — the usual cause of "dragging on" a topic.
        """
        messages = list(self.messages)
        if max_assistant is not None:
            max_assistant = max(1, max_assistant)  # keep the invariant local
            # Locate the max_assistant-th assistant message from the end and
            # keep everything from there onwards (including the user message
            # that prompted it, so the tail starts on a coherent pair).
            start = 0
            seen = 0
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "assistant":
                    seen += 1
                    if seen >= max_assistant:
                        if i > 0 and messages[i - 1]["role"] == "user":
                            start = i - 1
                        else:
                            start = i
                        break
            messages = messages[start:]
        return [{"role": "system", "content": system_prompt}, *messages]

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) for the /context command."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4


class UserRateLimiter:
    """Sliding-window per-user rate limiter.

    ``allow()`` records a hit and returns ``True`` only if fewer than ``rate``
    hits happened in the last ``per_seconds``. A ``rate`` of 0 disables the
    limiter entirely.

    ``_max_users`` caps how many users are tracked so a long-lived bot never
    grows unbounded; evicting a random user's window is harmless (it just
    resets their count).
    """

    _max_users = 10_000

    def __init__(self, rate: int, per_seconds: float) -> None:
        self.rate = rate
        self.per_seconds = per_seconds
        self._hits: dict[int, list[float]] = {}

    def allow(self, user_id: int) -> bool:
        if self.rate <= 0 or self.per_seconds <= 0:
            return True
        if user_id not in self._hits and len(self._hits) >= self._max_users:
            self._hits.pop(next(iter(self._hits)))
        now = time.monotonic()
        times = self._hits.setdefault(user_id, [])
        cutoff = now - self.per_seconds
        times[:] = [t for t in times if t > cutoff]
        if len(times) >= self.rate:
            return False
        times.append(now)
        return True


class AIAgentCog(commands.Cog, name="AI Agent"):
    """Conversational AI backed by the DeepSeek API."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config: Config = bot.config  # type: ignore[attr-defined]

        # AsyncOpenAI is fully asynchronous, so API calls never block the
        # Discord event loop. The SDK also retries transient failures
        # ``max_retries`` times and enforces the request timeout.
        self.client = AsyncOpenAI(
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            timeout=self.config.request_timeout_seconds,
            max_retries=self.config.max_retries,
        )
        # channel/thread id -> ConversationMemory
        self.memories: dict[int, ConversationMemory] = {}
        # channel/thread id -> asyncio.Lock, serialising prompts in that channel
        self._locks: dict[int, asyncio.Lock] = {}
        self._rate_limiter = UserRateLimiter(
            self.config.ask_cooldown_rate,
            self.config.ask_cooldown_period_seconds,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    # Hard cap on tracked channels so a long-lived bot never grows unbounded.
    # (Locks are tiny objects, so only the memory cache is capped.)
    CHANNEL_CACHE_MAX = 512

    def _evict_if_full(self, cache: dict) -> None:
        """Drop one arbitrary entry once the cache exceeds its cap."""
        if len(cache) >= self.CHANNEL_CACHE_MAX:
            cache.pop(next(iter(cache)))

    def _memory_for(self, channel_id: int) -> ConversationMemory:
        memory = self.memories.get(channel_id)
        if memory is None:
            self._evict_if_full(self.memories)
            memory = ConversationMemory(self.config.max_context_messages)
            self.memories[channel_id] = memory
        return memory

    def _lock_for(self, channel_id: int) -> asyncio.Lock:
        lock = self._locks.get(channel_id)
        if lock is None:
            if len(self._locks) >= self.CHANNEL_CACHE_MAX:
                # Evict only leaked locks: uncontended, and whose channel no
                # longer has a memory entry (e.g. after /clear). A channel with
                # a live conversation keeps its lock, preserving the per-channel
                # serialization invariant.
                for key, existing in self._locks.items():
                    if key not in self.memories and not existing.locked():
                        self._locks.pop(key)
                        break
            lock = asyncio.Lock()
            self._locks[channel_id] = lock
        return lock

    def _error_embed(self, title: str, description: str) -> discord.Embed:
        return discord.Embed(title=f"⚠️ {title}", description=description, color=discord.Color.red())

    def _prepare_messages(self, memory: ConversationMemory) -> list[dict]:
        """Build the system prompt (persona + guidance) plus trimmed history."""
        system = f"{self.config.system_prompt}\n\n{CONVERSATION_GUIDANCE}"
        if self.config.enable_tools:
            system += f"\n\n{TOOL_GUIDANCE}"
        return memory.to_api_messages(system, max_assistant=self.config.max_assistant_messages)

    async def _create_completion(self, messages: list[dict], *, stream: bool):
        """Call the DeepSeek API with the given messages."""
        return await self.client.chat.completions.create(
            model=self.config.deepseek_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=stream,
        )

    async def _ask_impl(
        self,
        sender,
        channel_id: int,
        prompt: str,
        *,
        author_id: int | None = None,
        author_name: str | None = None,
        reply_to: discord.Message | None = None,
    ) -> None:
        """Shared entry point for both commands and the auto-reply listener.

        Serialises per-channel access with a lock (so concurrent prompts in the
        same channel never interleave inside the conversation memory), enforces
        the per-user rate limit, and rolls the prompt back out of memory if the
        API call fails. The sender's username (``author_name``) is attached to
        the prompt so the model can tell who is speaking.

        ``sender`` is either a ``commands.Context`` or a ``discord.abc.Messageable``
        channel — both expose ``send()``; ``ctx`` additionally supports
        ``defer()``/``typing()`` for the slash-vs-prefix differences.
        """
        if author_id is not None and not self._rate_limiter.allow(author_id):
            embed = self._error_embed(
                "Slow down ⏳",
                f"You're asking too fast — max {self.config.ask_cooldown_rate} prompts "
                f"per {self.config.ask_cooldown_period_seconds:.0f}s. Please wait a moment "
                "and try again.",
            )
            await sender.send(embed=embed)
            return

        async with self._lock_for(channel_id):
            memory = self._memory_for(channel_id)
            memory.add_user(prompt, author_name)
            try:
                messages = self._prepare_messages(memory)
                if self.config.enable_tools:
                    await self._tool_answer(sender, memory, messages, reply_to=reply_to)
                elif self.config.stream_responses:
                    await self._stream_answer(sender, memory, messages, reply_to=reply_to)
                else:
                    await self._buffered_answer(sender, memory, messages, reply_to=reply_to)
            except Exception as exc:
                # Roll the unanswered prompt back out so a failed request never
                # leaves an orphan "user" message polluting the next context.
                if memory.messages and memory.messages[-1]["role"] == "user":
                    memory.messages.pop()
                await self._handle_api_error(sender, exc)

    async def _buffered_answer(self, sender, memory: ConversationMemory, messages: list[dict], *, reply_to=None) -> None:
        """Non-streaming path: show a typing indicator, then send the full reply."""
        # Interactions show a native "thinking" state (we deferred); prefix
        # commands and channel senders show the typing indicator instead.
        typing_cm = (
            nullcontext()
            if getattr(sender, "interaction", None) is not None
            else sender.typing()
        )
        async with typing_cm:
            completion = await self._create_completion(messages, stream=False)

        reply = (completion.choices[0].message.content or "").strip() or "*(no response)*"
        memory.add_assistant(reply)

        for i, chunk in enumerate(chunk_text(reply)):
            if i == 0 and reply_to is not None:
                await reply_to.reply(chunk)
            else:
                await sender.send(chunk)

    async def _stream_answer(self, sender, memory: ConversationMemory, messages: list[dict], *, reply_to=None) -> None:
        """Streaming path: live-edit a placeholder message as tokens arrive.

        Completed ~1900-char pieces are posted as their own messages; the
        placeholder is edited (throttled) to show progress. On a mid-stream
        failure any partial reply is kept and the error is surfaced after.
        """
        placeholder = await (
            reply_to.reply("🤔 *Thinking…*") if reply_to is not None else sender.send("🤔 *Thinking…*")
        )
        buffer = ""
        sent_chars = 0
        last_edit = 0.0
        last_edit_len = 0

        try:
            stream = await self._create_completion(messages, stream=True)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None or not delta.content:
                    continue
                buffer += delta.content

                # Push completed full-size pieces out as real messages.
                while len(buffer) - sent_chars > SAFE_MESSAGE_LIMIT:
                    piece = buffer[sent_chars : sent_chars + SAFE_MESSAGE_LIMIT]
                    sent_chars += len(piece)
                    await sender.send(piece)

                # Throttled live preview on the placeholder message.
                preview = buffer[sent_chars:]
                now = time.monotonic()
                if (
                    preview
                    and now - last_edit >= EDIT_THROTTLE_SECONDS
                    and len(preview) - last_edit_len >= EDIT_MIN_DELTA_CHARS
                ):
                    await placeholder.edit(content=f"{preview}…")
                    last_edit = now
                    last_edit_len = len(preview)
        except Exception:
            # Keep whatever partial reply we produced before failing.
            if buffer:
                await self._finalize_stream(placeholder, sender, buffer, sent_chars)
            raise

        reply = buffer.strip()
        if not reply:
            await placeholder.edit(content="*(no response)*")
            return

        memory.add_assistant(reply)
        await self._finalize_stream(placeholder, sender, buffer, sent_chars)

    @staticmethod
    async def _finalize_stream(placeholder: discord.Message, sender, buffer: str, sent_chars: int) -> None:
        """Deliver the tail of a streamed reply and replace the placeholder."""
        remaining = buffer[sent_chars:]
        chunks = chunk_text(remaining) or ["*(no response)*"]
        await placeholder.edit(content=chunks[0])
        for extra in chunks[1:]:
            await sender.send(extra)

    async def _tool_answer(self, sender, memory: ConversationMemory, messages: list[dict], *, reply_to=None) -> None:
        """Tool-calling path: run any requested tools, then send the buffered final reply."""
        placeholder = await (
            reply_to.reply("🤔 *Thinking…*") if reply_to is not None else sender.send("🤔 *Thinking…*")
        )
        try:
            messages, reply = await self._run_tool_loop(messages, placeholder)
        except Exception:
            # Don't leave the "thinking" placeholder stuck if the API died mid-loop.
            try:
                await placeholder.edit(content="⚠️ *(falhou ao usar ferramentas — erro acima)*")
            except discord.Forbidden:
                pass
            raise
        reply = reply.strip() or "*(no response)*"
        memory.add_assistant(reply)
        chunks = chunk_text(reply) or ["*(no response)*"]
        await placeholder.edit(content=chunks[0])
        for extra in chunks[1:]:
            await sender.send(extra)

    async def _run_tool_loop(self, messages: list[dict], placeholder: discord.Message | None) -> tuple[list[dict], str]:
        """Execute any tool calls the model requests, up to a safety cap.

        Appends the assistant tool-call turn and each tool result to
        ``messages`` (mutated in place), then returns ``(messages, final_text)``.
        The final answer is always non-streamed — tool calls add latency anyway.
        """
        schemas = agent_tools.tool_schemas(self.config.tenor_api_key)
        for _ in range(self.config.max_tool_iterations):
            completion = await self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False,
                tools=schemas,
            )
            message = completion.choices[0].message
            if not message.tool_calls:
                return messages, message.content or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                }
            )
            for tc in message.tool_calls:
                if placeholder is not None:
                    emoji = TOOL_STATUS_EMOJI.get(tc.function.name, "⚙️")
                    await placeholder.edit(content=f"{emoji} *{tc.function.name.replace('_', ' ')}…*")
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await agent_tools.run_tool(
                    tc.function.name,
                    args,
                    tenor_api_key=self.config.tenor_api_key,
                    max_chars=self.config.page_fetch_max_chars,
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Safety cap reached — force a final answer without tools.
        completion = await self.client.chat.completions.create(
            model=self.config.deepseek_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=False,
        )
        return messages, completion.choices[0].message.content or ""

    async def _handle_api_error(self, sender, exc: Exception) -> None:
        """Map API/Discord exceptions to a friendly embed."""
        log.error("DeepSeek API call failed: %s", exc, exc_info=exc)

        if isinstance(exc, AuthenticationError):
            embed = self._error_embed(
                "Invalid API key",
                "The DeepSeek API key is invalid or revoked. Check `DEEPSEEK_API_KEY` in your `.env` file.",
            )
        elif isinstance(exc, RateLimitError):
            embed = self._error_embed(
                "Rate limited",
                "DeepSeek is rate-limiting requests right now. Please wait a moment and try again.",
            )
        elif isinstance(exc, APITimeoutError):
            embed = self._error_embed(
                "Request timed out",
                f"The DeepSeek API did not respond within {self.config.request_timeout_seconds:.0f}s. Try again.",
            )
        elif isinstance(exc, APIConnectionError):
            embed = self._error_embed(
                "Connection error",
                "Could not reach the DeepSeek API. Check the bot host's network connection.",
            )
        elif isinstance(exc, APIError):
            embed = self._error_embed(
                "DeepSeek API error",
                "The DeepSeek API returned an unexpected error. Check the bot logs for details.",
            )
        elif isinstance(exc, discord.Forbidden):
            embed = self._error_embed(
                "Missing permissions",
                "I don't have permission to send messages here. Ask an admin to grant me the `Send Messages` permission.",
            )
        else:
            embed = self._error_embed(
                "Unexpected error",
                "Something went wrong while generating a reply. Check the bot logs for details.",
            )

        try:
            await sender.send(embed=embed)
        except discord.Forbidden:
            log.warning("Could not send error embed (missing permissions) in %s", getattr(sender, "id", "?"))

    # ------------------------------------------------------------------ #
    # Message listener (auto-reply on mention / DM / designated channel)  #
    # ------------------------------------------------------------------ #

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # Never respond to bots (including ourselves) or webhooks.
        if message.author.bot or message.webhook_id:
            return

        # Prefix commands are handled by the command framework; skip them here
        # so ``!ask …`` doesn't trigger a duplicate auto-reply.
        if message.content.startswith(self.config.bot_prefix):
            return

        is_dm = message.guild is None
        mentioned = self.bot.user is not None and any(u.id == self.bot.user.id for u in message.mentions)
        in_allowed = message.channel.id in self.config.allowed_channels

        if not is_dm and not mentioned and not in_allowed:
            return

        prompt = _strip_mentions(message.content, self.bot.user.id).strip()
        if not prompt:
            return

        # Use the global username (author.name), not the server nickname
        # (display_name): the persona rules key on usernames, and nicknames
        # with spaces/unicode would all hit the content-prefix fallback.
        await self._ask_impl(
            message.channel,
            message.channel.id,
            prompt,
            author_id=message.author.id,
            author_name=message.author.name,
            reply_to=message,
        )

    # ------------------------------------------------------------------ #
    # Commands (hybrid = slash + prefix)                                  #
    # ------------------------------------------------------------------ #

    @commands.hybrid_command(
        name="ask",
        aliases=["chat", "a"],
        description="Ask DeepSeek a question (remembers this channel's context).",
    )
    async def ask(self, ctx: commands.Context, *, prompt: str) -> None:
        """Usage: ``/ask <prompt>`` or ``!ask <prompt>``."""
        # Slash invocations need an explicit defer to show the "thinking" state
        # and allow follow-up messages after the 3s interaction window.
        if ctx.interaction is not None:
            await ctx.defer()

        await self._ask_impl(
            ctx,
            ctx.channel.id,
            prompt,
            author_id=ctx.author.id,
            author_name=ctx.author.name,
        )

    @commands.hybrid_command(
        name="clear",
        aliases=["reset"],
        description="Forget this channel's conversation history.",
    )
    async def clear(self, ctx: commands.Context) -> None:
        """Usage: ``/clear`` or ``!clear``."""
        self.memories.pop(ctx.channel.id, None)  # drop the entry entirely, not just its contents
        embed = discord.Embed(
            title="🧹 Memory cleared",
            description="I've forgotten our previous conversation in this channel/thread.",
            color=discord.Color.green(),
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(
        name="context",
        description="Show conversation-context stats for this channel.",
    )
    async def context(self, ctx: commands.Context) -> None:
        """Usage: ``/context`` or ``!context``."""
        memory = self._memory_for(ctx.channel.id)
        embed = discord.Embed(title="🧠 Conversation Context", color=discord.Color.blue())
        embed.add_field(name="Channel / Thread", value=ctx.channel.mention, inline=True)
        embed.add_field(name="Stored messages", value=str(len(memory.messages)), inline=True)
        embed.add_field(name="Est. tokens", value=f"~{memory.estimated_tokens}", inline=True)
        embed.add_field(name="Model", value=f"`{self.config.deepseek_model}`", inline=True)
        embed.add_field(
            name="Streaming",
            value="✅ on" if self.config.stream_responses else "❌ off",
            inline=True,
        )
        embed.add_field(name="Max context", value=f"{self.config.max_context_messages} messages", inline=True)
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="ping", description="Check whether the bot is alive.")
    async def ping(self, ctx: commands.Context) -> None:
        """Usage: ``/ping`` or ``!ping``."""
        await ctx.send("Pong! 🏓")

    @commands.hybrid_command(name="help", description="Show available commands and usage.")
    async def help_command(self, ctx: commands.Context) -> None:
        """Custom help: lists every command plus the auto-reply behaviour."""
        embed = discord.Embed(
            title="🤖 DeepSeek Bot Help",
            color=discord.Color.blue(),
            description=f"Prefix: `{self.config.bot_prefix}` — every command also works as a slash command.",
        )
        embed.add_field(
            name=f"`/ask <prompt>`  ·  `{self.config.bot_prefix}ask <prompt>`",
            value="Ask the AI anything. Conversation history is kept per channel/thread.",
            inline=False,
        )
        embed.add_field(
            name=f"`/clear`  ·  `{self.config.bot_prefix}clear`",
            value="Forget the conversation history in this channel/thread.",
            inline=False,
        )
        embed.add_field(
            name=f"`/context`  ·  `{self.config.bot_prefix}context`",
            value="Show how much context is being tracked here.",
            inline=False,
        )
        embed.add_field(
            name=f"`/ping`  ·  `{self.config.bot_prefix}ping`",
            value="Health check.",
            inline=False,
        )
        embed.add_field(
            name="💬 Auto-reply",
            value=(
                "Mention the bot (`@DeepSeek <message>`) anywhere, or simply talk in a "
                "channel/thread listed in `ALLOWED_CHANNELS`. DMs are answered automatically."
            ),
            inline=False,
        )
        await ctx.send(embed=embed)

    # ------------------------------------------------------------------ #
    # Command-level error handling                                        #
    # ------------------------------------------------------------------ #

    async def cog_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        """Friendly embeds for user-facing command errors (missing args, etc.)."""
        if isinstance(error, commands.MissingRequiredArgument):
            usage = f"`{ctx.prefix}{ctx.command.qualified_name} <prompt>`" if ctx.command else "see /help"
            embed = self._error_embed("Missing argument", f"Usage: {usage}")
        elif isinstance(error, commands.BadArgument):
            embed = self._error_embed("Invalid argument", str(error))
        else:
            log.error("Command %s failed: %s", ctx.command, error, exc_info=error)
            embed = self._error_embed("Command error", "Something went wrong running that command.")
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """discord.py extension entry point."""
    await bot.add_cog(AIAgentCog(bot))
