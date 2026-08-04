"""Balance cog — shows the remaining DeepSeek API credit.

Queries DeepSeek's ``GET /user/balance`` endpoint (documented at
https://api-docs.deepseek.com/api/get-user-balance) and renders the result as
an embed, so you can check your remaining credit from inside Discord with
``/balance`` or ``!balance``.

Uses ``httpx`` directly because the balance endpoint is a DeepSeek-specific
route that the ``openai`` SDK does not expose. ``httpx`` is already installed
as a dependency of ``openai``, but it is pinned explicitly in
``requirements.txt`` so the import never breaks.
"""

from __future__ import annotations

import logging

import discord
import httpx
from discord.ext import commands

from config import Config

log = logging.getLogger(__name__)

BALANCE_TIMEOUT_SECONDS = 10.0


class BalanceCog(commands.Cog, name="Balance"):
    """Live DeepSeek credit tracking."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config: Config = bot.config  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _error_embed(self, title: str, description: str) -> discord.Embed:
        return discord.Embed(title=f"⚠️ {title}", description=description, color=discord.Color.red())

    async def _fetch_balance(self) -> dict:
        """GET /user/balance with the configured API key (fully async)."""
        url = f"{self.config.deepseek_base_url.rstrip('/')}/user/balance"
        headers = {"Authorization": f"Bearer {self.config.deepseek_api_key}"}
        async with httpx.AsyncClient(timeout=BALANCE_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _send_http_error(self, ctx: commands.Context, exc: httpx.HTTPStatusError) -> None:
        """Friendly embed for non-2xx responses from the balance endpoint."""
        status = exc.response.status_code
        if status == 401:
            embed = self._error_embed(
                "Invalid API key",
                "DeepSeek rejected the API key. Check `DEEPSEEK_API_KEY` in your `.env` file.",
            )
        elif status == 402:
            embed = self._error_embed(
                "Out of credit",
                "Your DeepSeek balance is exhausted — top up at platform.deepseek.com.",
            )
        else:
            embed = self._error_embed("Balance API error", f"DeepSeek returned HTTP {status}.")
        await ctx.send(embed=embed)

    # ------------------------------------------------------------------ #
    # Commands                                                            #
    # ------------------------------------------------------------------ #

    @commands.hybrid_command(
        name="balance",
        aliases=["credits"],
        description="Show your remaining DeepSeek API credit.",
    )
    async def balance(self, ctx: commands.Context) -> None:
        """Usage: ``/balance`` or ``!balance`` — shows live credit from DeepSeek."""
        # Slash invocations need a defer to allow follow-up messages.
        if ctx.interaction is not None:
            await ctx.defer()

        try:
            data = await self._fetch_balance()
        except httpx.HTTPStatusError as exc:
            await self._send_http_error(ctx, exc)
            return
        except (httpx.ConnectError, httpx.TimeoutException):
            embed = self._error_embed(
                "Connection error",
                "Could not reach the DeepSeek balance API right now. Try again shortly.",
            )
            await ctx.send(embed=embed)
            return

        available = bool(data.get("is_available", True))
        infos = data.get("balance_infos") or []

        if not infos:
            embed = self._error_embed(
                "No balance data",
                "DeepSeek returned no balance information for this API key.",
            )
            await ctx.send(embed=embed)
            return

        embed = discord.Embed(
            title="💰 DeepSeek Credit Balance",
            color=discord.Color.green() if available else discord.Color.red(),
        )
        for info in infos:
            currency = info.get("currency", "?")
            embed.add_field(name=f"Total ({currency})", value=f"**{info.get('total_balance', '0.00')}**", inline=True)
            embed.add_field(name="Granted", value=info.get("granted_balance", "0.00"), inline=True)
            embed.add_field(name="Topped up", value=info.get("topped_up_balance", "0.00"), inline=True)

        if not available:
            embed.add_field(
                name="⚠️ Account status",
                value="DeepSeek reports the account as **not available for API calls** — "
                      "your balance may be exhausted or the account restricted.",
                inline=False,
            )
        embed.set_footer(text=f"Model: {self.config.deepseek_model} · live from DeepSeek")
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """discord.py extension entry point."""
    await bot.add_cog(BalanceCog(bot))
