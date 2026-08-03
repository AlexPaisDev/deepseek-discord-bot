# 🤖 DeepSeek Discord Bot

A production-ready Discord bot that acts as an AI conversational agent powered by the
**DeepSeek API** (`api.deepseek.com`), built with `discord.py` v2.x.

It supports **slash commands and prefix commands**, responds when **@mentioned**, listens
in **designated channels/threads**, keeps **per-channel conversation memory**, and
**streams replies** live — all without ever blocking Discord's event loop.

---

## ✨ Features

| Feature | How it works |
|---|---|
| Slash + prefix commands | One `@commands.hybrid_command` definition powers `/ask` and `!ask` |
| Multi-channel / thread support | Auto-replies in channels/threads listed in `ALLOWED_CHANNELS`, or anywhere when @mentioned |
| Chat memory | Rolling context per channel/thread (`deque` capped at `MAX_CONTEXT_MESSAGES`) — old messages drop off automatically to save tokens |
| Streaming + typing indicator | Live-edits a "Thinking…" placeholder as tokens arrive (throttled to respect rate limits); falls back to a typing indicator when streaming is off |
| Long-message splitting | Replies are chunked into Discord-safe ≤1900-char messages |
| Friendly error handling | Rate limits, timeouts, invalid keys, and permission errors surface as clean embeds instead of crashes |
| Fully async | `AsyncOpenAI` client + `async`/`await` everywhere — the event loop is never blocked |
| 24/7 hosting | Ready-made `Dockerfile` + `docker-compose.yml` |

---

## 📁 Project structure

```
.
├── bot.py               # Entry point: intents, on_ready, slash-command sync
├── config.py            # Typed env-var loader (python-dotenv)
├── cogs/
│   ├── __init__.py
│   └── ai_agent.py      # All DeepSeek calls, memory, streaming, commands
├── .env.example         # Template — copy to .env and fill in
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Setup

### 1. Prerequisites

- Python **3.10+** (tested on 3.11)
- A Discord application with a bot account
- A DeepSeek API key from [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)

### 2. Create the Discord application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application** → give it a name (e.g. `DeepSeek`).
2. In the left sidebar open **Bot** and click **Add Bot**.
3. Under **Bot → Token**, click **Reset Token** and copy it — this is your `DISCORD_TOKEN`. **Treat it like a password; never commit it.**
4. Under **Privileged Gateway Intents**, enable **both**:
   - ✅ **Message Content Intent** — required to read message text for commands and prompts
   - ✅ **Server Members Intent** — required for member data (used by the bot's intents)

   > ⚠️ For bots in more than 100 servers you must additionally verify your app (not needed for testing).

5. Open the **OAuth2 → URL Generator** tab and generate an invite link:
   - **Scopes:** `bot`, `applications.commands`
   - **Bot Permissions:** `Send Messages`, `Read Messages / View Channels`, `Read Message History`, `Embed Links`, `Use Slash Commands` (or just grant **Administrator** for a private server)
   - Copy the generated URL and open it in a browser to add the bot to your server.

### 3. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```ini
DISCORD_TOKEN=your-bot-token-here
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### 4. Run locally (for testing)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python bot.py

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

You should see logs like:

```
INFO bot: Logged in as DeepSeek#1234 (ID 123456789) in 1 guild(s)
INFO bot: Synced 5 slash command(s)
```

Test it in Discord: `/ask hello` or `!ask hello`, or just **@mention the bot**.

### No Python installed on your PC? 🤷

You don't need it — this repo ships with a **GitHub Actions CI workflow** that
installs Python in the cloud and verifies the code (syntax, imports, config
validation, chunking unit tests) on every push. Just push to GitHub and watch
the **Actions** tab go green. For actually *running* the bot, use the cloud
deploy options below (Railway / Render) — zero local installs.

---

## ⚙️ Configuration reference

All settings are environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `DISCORD_TOKEN` | — (required) | Bot token from the Developer Portal |
| `DEEPSEEK_API_KEY` | — (required) | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model name. If your account doesn't recognise it, use `deepseek-chat` |
| `BOT_PREFIX` | `!` | Prefix for text commands |
| `BOT_STATUS` | `DeepSeek V4 Flash — /ask or @mention me` | Status line under the bot's name |
| `ALLOWED_CHANNELS` | *(empty)* | Comma-separated channel/thread IDs where the bot answers every message (no mention needed). Enable Developer Mode → right-click channel → **Copy Channel ID** |
| `MAX_CONTEXT_MESSAGES` | `20` | Rolling history kept per channel/thread |
| `MAX_TOKENS` | `2048` | Max tokens per reply |
| `TEMPERATURE` | `0.7` | Creativity (0.0 strict → 1.0 creative) |
| `REQUEST_TIMEOUT_SECONDS` | `60` | DeepSeek request timeout |
| `MAX_RETRIES` | `2` | SDK auto-retries for transient API failures |
| `STREAM_RESPONSES` | `true` | Live-edit streaming replies; set `false` for typing-indicator + buffered replies |
| `SYSTEM_PROMPT` | built-in persona | Override the AI's system prompt |

---

## 💬 Usage

| Command | Description |
|---|---|
| `/ask <prompt>` · `!ask <prompt>` | Ask the AI (keeps this channel's context) |
| `/clear` · `!clear` | Forget this channel's conversation history |
| `/context` · `!context` | Show stored messages / estimated tokens for this channel |
| `/ping` · `!ping` | Health check |
| `/help` · `!help` | List commands and behaviour |

**Auto-reply modes:**

- **@mention:** `@DeepSeek what is a quaternion?` — works in any channel/thread.
- **Designated channels/threads:** add their IDs to `ALLOWED_CHANNELS` and every message is answered.
- **DMs:** the bot answers every DM automatically.

> 💡 **Thread tip:** to keep a topic's context isolated, create a thread for it and add the
> thread's ID to `ALLOWED_CHANNELS` — context is tracked **per channel/thread**, so parallel
> conversations never bleed into each other.

---

## 🖥️ Deploy for 24/7 hosting

### Option A — Docker on any VPS (recommended)

On the VPS (requires [Docker](https://docs.docker.com/engine/install/)):

```bash
git clone <your-repo-url> && cd discord_bot
cp .env.example .env          # fill in your real token + API key
docker compose up -d --build  # starts and auto-restarts on crashes/reboots
```

Useful commands:

```bash
docker compose logs -f          # watch the logs
docker compose restart          # after editing .env
docker compose pull && docker compose up -d --build   # deploy a new version
docker compose down             # stop
```

`restart: unless-stopped` keeps the bot alive across crashes and reboots automatically.

### Option B — Railway

1. Push this repo to GitHub.
2. In [Railway](https://railway.app) click **New Project → Deploy from GitHub repo**.
3. Railway auto-detects the `Dockerfile`. Add the two variables under **Variables**:
   - `DISCORD_TOKEN=…`
   - `DEEPSEEK_API_KEY=…`
4. Deploy. Railway restarts the service automatically if it crashes.

### Option C — Render

1. Push the repo to GitHub.
2. In [Render](https://render.com) click **New → Background Worker** and connect the repo.
3. Set **Build Command** to `pip install -r requirements.txt` and **Start Command** to `python bot.py`.
4. Add the environment variables `DISCORD_TOKEN` and `DEEPSEEK_API_KEY` under **Environment**.
5. Create the service — it runs continuously and auto-restarts on failure.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Configuration error: DISCORD_TOKEN is required` | You forgot to create `.env` from `.env.example` |
| `Privileged intents …` warning at startup | Enable **Message Content Intent** in the Developer Portal |
| Slash commands don't appear | Re-invite the bot with the `applications.commands` scope; global slash commands can take up to an hour to propagate (restarting the bot resyncs) |
| `Invalid API key` embed | Double-check `DEEPSEEK_API_KEY` |
| Model not found | Set `DEEPSEEK_MODEL=deepseek-chat` if `deepseek-v4-flash` isn't available on your account |
| Bot doesn't reply in a channel | Add the channel ID to `ALLOWED_CHANNELS` or mention the bot |

---

## 🔒 Notes on production

- **Never commit `.env`** — it's already in `.gitignore`.
- Context is held **in memory** and resets when the bot restarts. For persistent
  long-term memory, wire a database (e.g. Redis/SQLite) into `ConversationMemory`.
- `MAX_CONTEXT_MESSAGES` bounds token usage per request; combined with the rolling
  deque, context costs stay predictable.
- The OpenAI SDK's built-in retries (`MAX_RETRIES`) and the request timeout handle
  transient API flakiness; anything else surfaces as a friendly embed.
