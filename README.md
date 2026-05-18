# TradingAgents — Web Frontend + Codex OAuth

A fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) that adds:

1. **Codex OAuth provider** — run the full agent pipeline using your ChatGPT/Codex OAuth session instead of a paid OpenAI API key.
2. **Web frontend** — a browser-based UI with live agent progress, BUY/SELL/HOLD decision display, and expandable analyst reports.

---

## How it works

The `openai-codex` provider reads an OAuth access token from [Hermes](https://github.com/dillionverma/hermes) (`~/.hermes/auth.json`) and forwards requests to the ChatGPT Codex backend (`https://chatgpt.com/backend-api/codex`). No OpenAI API key or billing account is required — it uses your existing ChatGPT session.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A ChatGPT account with Codex access
- Hermes CLI authenticated with `openai-codex`

---

## Installation

```bash
git clone https://github.com/geoff-rose/TradingAgents-Web-Codex-OAuth
cd TradingAgents-Web-Codex-OAuth
uv sync
```

---

## Codex OAuth Setup

### 1. Authenticate with Hermes

If you haven't already, install Hermes and log in:

```bash
hermes auth add openai-codex
```

This stores an OAuth access token at `~/.hermes/auth.json`. The token is read automatically — no extra configuration needed.

### 2. Token lookup order

The client tries the following in order:

| Source | How to set |
|--------|-----------|
| Environment variable | `export TRADINGAGENTS_CODEX_ACCESS_TOKEN=<token>` |
| Hermes auth file (default) | `~/.hermes/auth.json` |
| Custom Hermes path | `export HERMES_AUTH_FILE=/path/to/auth.json` |

The environment variable takes priority, which is useful in CI or Docker environments where you want to inject the token directly.

### 3. auth.json structure

Hermes stores credentials in one of two layouts — both are supported:

**Credential pool layout** (newer Hermes versions):
```json
{
  "credential_pool": {
    "openai-codex": [
      { "access_token": "eyJ..." }
    ]
  }
}
```

**Providers layout** (older Hermes versions):
```json
{
  "providers": {
    "openai-codex": {
      "tokens": {
        "access_token": "eyJ..."
      }
    }
  }
}
```

If you manage tokens manually (without Hermes), create a file in either format at `~/.hermes/auth.json`, or set `TRADINGAGENTS_CODEX_ACCESS_TOKEN` directly.

### 4. Available models

When selecting `openai-codex` as the provider, the following models are available:

| Mode | Models |
|------|--------|
| Quick | `gpt-5.5`, `gpt-5.4-mini` |
| Deep  | `gpt-5.5`, `gpt-5.4`, `gpt-5.3-codex-spark` |

Model availability depends on your account entitlements. Use **Custom model ID** in the CLI to specify any model not listed.

---

## CLI Usage

Launch the standard interactive CLI:

```bash
python -m cli.main
```

Select **openai-codex** as the LLM provider when prompted. The token is loaded automatically from Hermes — no API key entry required.

---

## Web Frontend

### Start the server

```bash
uv run python -m uvicorn web.server:app --host 0.0.0.0 --port 7777
```

Then open **http://localhost:7777** in your browser.

To access from other devices on your local network, use your machine's LAN IP:

```
http://192.168.x.x:7777
```

### Using the UI

| Field | Description |
|-------|-------------|
| **Ticker** | Stock symbol, e.g. `AAPL`, `TSLA`, `BET.L` |
| **Analysis Date** | The date to run the analysis for |
| **Research Depth** | Shallow (1 debate round) · Medium (3) · Deep (5) |
| **Quick Model** | Used for fast analyst and researcher nodes |
| **Deep Model** | Used for the trader and portfolio manager nodes |
| **Analysts** | Toggle individual analyst agents on/off |

Click **Analyse** to start. The agent pipeline streams live progress as each agent completes. When finished, the final **BUY / SELL / HOLD** decision is shown at the top, with expandable panels for each analyst report below.

### Architecture

```
Browser  ──POST /api/analyze──►  FastAPI (web/server.py)
         ◄──GET /api/stream/──   │
              (SSE)              │  background thread
                                 ▼
                        TradingAgentsGraph.propagate()
                        (openai-codex provider, Hermes token)
```

The server runs the agent graph in a thread pool and streams node-completion events to the browser via Server-Sent Events. Each event corresponds to one agent finishing its work.

### Firewall

If running on Linux with ufw, allow the port:

```bash
sudo ufw allow 7777
```

---

## Configuration

All standard `TRADINGAGENTS_*` environment variables are supported. Set them in `.env` or export them before starting the server:

```bash
# Use a different default model
export TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
export TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini

# Adjust debate depth globally
export TRADINGAGENTS_MAX_DEBATE_ROUNDS=3

# Point Hermes at a non-default location
export HERMES_AUTH_FILE=/custom/path/auth.json
```

---

## Project structure

```
├── tradingagents/llm_clients/
│   ├── openai_codex_client.py   # Codex OAuth provider (new)
│   ├── model_catalog.py         # Added openai-codex model list
│   ├── factory.py               # Routes openai-codex to the new client
│   ├── api_key_env.py           # Skips API key requirement for OAuth
│   └── validators.py            # Skips key validation for openai-codex
├── cli/utils.py                 # CLI support for openai-codex provider
└── web/
    ├── server.py                # FastAPI backend with SSE streaming
    └── static/
        └── index.html           # Single-page frontend
```

---

## Credits

Built on top of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).  
OAuth integration uses the [Hermes](https://github.com/dillionverma/hermes) credential store.
