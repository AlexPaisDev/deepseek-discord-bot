"""DeepSeek Discord bot — main entry point.

Initialises the :class:`~discord.ext.commands.Bot` with the required intents,
loads the :mod:`cogs.ai_agent` extension (which owns all DeepSeek interaction)
and wires up the core lifecycle events:

* ``on_ready``  — logs in, sets presence, syncs slash commands.
* ``on_message`` — feeds every message into the command framework (prefix
  commands). The AI auto-reply logic lives in the cog's own listener, so the
  two behaviours never interfere.

Run with: ``python bot.py``
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from config import Config, ConfigError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")


class DeepSeekBot(commands.Bot):
    """The bot instance. Carries the validated :class:`Config` for cogs."""

    def __init__(self, config: Config) -> None:
        self.config = config

        # Intents required for this bot to function:
        #  - message_content: read message text (prefix commands + AI prompts)
        #  - members:        Server Members Intent (guild member data)
        # Both must be enabled in the Developer Portal (see README).
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix=config.bot_prefix,
            intents=intents,
            help_command=None,  # we ship a custom /help in the cog
        )

    async def setup_hook(self) -> None:
        """Async setup: load extensions before the bot connects."""
        await self.load_extension("cogs.ai_agent")
        await self.load_extension("cogs.balance")
        log.info("Loaded cogs.ai_agent, cogs.balance")

    async def on_ready(self) -> None:
        """Fired once when the websocket connection is ready."""
        assert self.user is not None
        log.info("Logged in as %s (ID %s) in %d guild(s)", self.user, self.user.id, len(self.guilds))

        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=self.config.bot_status,
        )
        await self.change_presence(activity=activity)

        # Hybrid commands are registered on the command tree at load time, but
        # Discord needs an explicit sync before they appear as slash commands.
        try:
            synced = await self.tree.sync()
            log.info("Synced %d slash command(s)", len(synced))
        except Exception:
            log.exception("Failed to sync slash commands (will retry on next restart)")

    async def on_message(self, message: discord.Message) -> None:
        """Route prefix-command messages into the command framework.

        Overriding this is the documented way to keep control; the AI
        auto-reply listener in the cog runs independently of this.
        """
        await self.process_commands(message)

    async def on_command_error(self, context: commands.Context, error: commands.CommandError) -> None:
        """Global fallback for errors not handled inside a cog."""
        if isinstance(error, commands.CommandNotFound):
            return  # unknown prefix commands are silent — the AI listener handles them
        log.error("Unhandled command error in %s: %s", context.command, error, exc_info=error)


async def main() -> None:
    try:
        config = Config.from_env()
    except ConfigError as exc:
        log.critical("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    bot = DeepSeekBot(config)
    async with bot:
        await bot.start(config.discord_token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
