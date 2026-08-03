"""AI Agent cog — all DeepSeek API interaction lives here.

Responsibilities:
  * Slash + prefix commands (``ask``, ``clear``, ``context``, ``ping``, ``help``)
    built with :func:`discord.ext.commands.hybrid_command` so one definition
    powers both interfaces.
  * An ``on_message`` listener that auto-replies when the bot is @mentioned,
    messaged in DMs, or spoken to in a designated channel/thread.
  * Per-channel/thread rolling conversation memory (a bounded deque) so context
    stays fresh without burning tokens.
  * Streaming responses with live message edits (throttled to respect Discord
    rate limits), falling back to a typing indicator when streaming is off.
  * Friendly error embeds for rate limits, timeouts, invalid keys, permissions.
  * Splitting of long replies into Discord-safe <=1900-char chunks.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

# Discord hard limit is 2000 chars per message; keep a safety margin.
SAFE_MESSAGE_LIMIT = 1900
# Live-edit throttling: stay comfortably under Discord's ~5 edits / 5s per channel.
EDIT_THROTTLE_SECONDS = 0.9
EDIT_MIN_DELTA_CHARS = 100


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
    """

    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self.messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.messages.clear()

    def to_api_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Messages ready to send to the API (system prompt + rolling history)."""
        return [{"role": "system", "content": system_prompt}, *self.messages]

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) for the /context command."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4


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

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _memory_for(self, channel_id: int) -> ConversationMemory:
        memory = self.memories.get(channel_id)
        if memory is None:
            memory = ConversationMemory(self.config.max_context_messages)
            self.memories[channel_id] = memory
        return memory

    def _error_embed(self, title: str, description: str) -> discord.Embed:
        return discord.Embed(title=f"⚠️ {title}", description=description, color=discord.Color.red())

    async def _create_completion(self, memory: ConversationMemory, *, stream: bool):
        """Call the DeepSeek API with the channel's rolling context."""
        messages = memory.to_api_messages(self.config.system_prompt)
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
        memory: ConversationMemory,
        prompt: str,
        *,
        reply_to: discord.Message | None = None,
    ) -> None:
        """Shared entry point for both commands and the auto-reply listener.

        ``sender`` is either a ``commands.Context`` or a ``discord.abc.Messageable``
        channel — both expose ``send()``; ``ctx`` additionally supports
        ``defer()``/``typing()`` for the slash-vs-prefix differences.
        """
        memory.add_user(prompt)
        try:
            if self.config.stream_responses:
                await self._stream_answer(sender, memory, reply_to=reply_to)
            else:
                await self._buffered_answer(sender, memory, reply_to=reply_to)
        except Exception as exc:
            await self._handle_api_error(sender, exc)

    async def _buffered_answer(self, sender, memory: ConversationMemory, *, reply_to=None) -> None:
        """Non-streaming path: show a typing indicator, then send the full reply."""
        # Interactions show a native "thinking" state (we deferred); prefix
        # commands and channel senders show the typing indicator instead.
        typing_cm = (
            nullcontext()
            if getattr(sender, "interaction", None) is not None
            else sender.typing()
        )
        async with typing_cm:
            completion = await self._create_completion(memory, stream=False)

        reply = (completion.choices[0].message.content or "").strip() or "*(no response)*"
        memory.add_assistant(reply)

        for i, chunk in enumerate(chunk_text(reply)):
            if i == 0 and reply_to is not None:
                await reply_to.reply(chunk)
            else:
                await sender.send(chunk)

    async def _stream_answer(self, sender, memory: ConversationMemory, *, reply_to=None) -> None:
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
            stream = await self._create_completion(memory, stream=True)
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
            embed = self._error_embed("DeepSeek API error", f"The API returned an error: {exc}")
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

        memory = self._memory_for(message.channel.id)
        await self._ask_impl(message.channel, memory, prompt, reply_to=message)

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

        memory = self._memory_for(ctx.channel.id)
        await self._ask_impl(ctx, memory, prompt)

    @commands.hybrid_command(
        name="clear",
        aliases=["reset"],
        description="Forget this channel's conversation history.",
    )
    async def clear(self, ctx: commands.Context) -> None:
        """Usage: ``/clear`` or ``!clear``."""
        self._memory_for(ctx.channel.id).clear()
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
