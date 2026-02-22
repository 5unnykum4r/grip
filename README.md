<div align="center">
  <h1>grip-ai</h1>
  <p>Claude Agent SDK powered AI platform — self-hostable, multi-model fallback via LiteLLM, multi-channel.</p>
  <p>
    <a href="#installation"><strong>Install</strong></a> &nbsp;&middot;&nbsp;
    <a href="#quickstart"><strong>Quickstart</strong></a> &nbsp;&middot;&nbsp;
    <a href="#telegram-setup"><strong>Telegram</strong></a> &nbsp;&middot;&nbsp;
    <a href="#api-reference"><strong>API</strong></a> &nbsp;&middot;&nbsp;
    <a href="#configuration"><strong>Config</strong></a>
  </p>
</div>

<p align="center">
  <a href="https://github.com/5unnykum4r/grip-ai/actions/workflows/ci.yml"><img src="https://github.com/5unnykum4r/grip-ai/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/grip-ai/"><img src="https://img.shields.io/pypi/v/grip-ai.svg" alt="PyPI Version"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/engine-Claude%20Agent%20SDK-blueviolet" alt="Claude Agent SDK">
  <img src="https://img.shields.io/badge/providers-15-orange" alt="15 LLM Providers">
</p>

---

grip is a self-hostable AI agent platform built with Python and [uv](https://docs.astral.sh/uv/). It uses the **Claude Agent SDK** as its primary engine for Claude models, with a **LiteLLM fallback** for 15+ other providers (OpenAI, DeepSeek, Groq, Gemini, Ollama local & cloud, etc.). Chat over Telegram/Discord/Slack, schedule cron tasks, orchestrate multi-agent workflows, and expose a secure REST API — all from a single process.

## Features

| Category | Details |
|----------|---------|
| **Dual Engine** | Claude Agent SDK (primary, recommended) + LiteLLM fallback for non-Claude models |
| **LLM Providers** | Anthropic (via SDK), OpenRouter, OpenAI, DeepSeek, Groq, Google Gemini, Qwen, MiniMax, Moonshot (Kimi), Ollama (Cloud), Ollama (Local), vLLM, Llama.cpp, LM Studio, and any OpenAI-compatible API |
| **Built-in Tools** | File read/write/edit/append/list/delete, shell execution, web search (Brave + DuckDuckGo), web fetch, messaging, subagent spawning, finance (yfinance), MCP tools |
| **Chat Channels** | Telegram (bot commands, photos, documents, voice), Discord, Slack (Socket Mode) |
| **REST API** | FastAPI with bearer auth, rate limiting, audit logging, security headers, 21 endpoints |
| **Workflows** | DAG-based multi-agent orchestration with dependency resolution and parallel execution |
| **Memory** | Dual-layer (MEMORY.md + HISTORY.md) with TF-IDF retrieval, auto-consolidation, and semantic caching |
| **Scheduling** | Cron jobs with channel delivery, heartbeat service, natural language scheduling |
| **Skills** | 18 built-in markdown skills, workspace overrides, install/remove via CLI |
| **Security** | Directory trust model, shell deny-list (50+ patterns), secret sanitizer, Shield runtime threat feed policy, token tracking, rate limiting |
| **Observability** | OpenTelemetry tracing, in-memory metrics, crash recovery, config validation |

## Engine Modes

grip uses a dual-engine architecture controlled by the `engine` config field:

| Engine | Config Value | Use Case |
|--------|-------------|----------|
| **Claude Agent SDK** | `claude_sdk` (default) | Anthropic Claude models — full agentic loop, tool execution, and context management handled by the SDK |
| **LiteLLM** | `litellm` | Non-Claude models (OpenAI, DeepSeek, Groq, Gemini, Ollama Cloud/Local, etc.) — uses grip's internal agent loop with LiteLLM for API calls |

Switch engines via config:

```bash
# Use Claude Agent SDK (default)
grip config set agents.defaults.engine "claude_sdk"

# Use LiteLLM for non-Claude models
grip config set agents.defaults.engine "litellm"
```

The onboarding wizard (`grip onboard`) automatically sets the right engine based on your provider choice.

## <a id="installation"></a>Installation

### Recommended: Via PyPI

```bash
# Using uv (faster)
uv tool install grip-ai

# Using pip
pip install grip-ai
```

### Manual Install (from source)

```bash
# Clone the repository
git clone https://github.com/5unnykum4r/grip-ai.git
cd grip

# Install (includes Telegram, REST API, and all core features)
uv sync

# Optional extras
uv sync --extra discord      # Discord bot
uv sync --extra slack        # Slack bot (Socket Mode)
uv sync --extra mcp          # Model Context Protocol
uv sync --extra finance      # Financial tools (yfinance)
uv sync --extra viz          # Data visualization (plotext)
uv sync --extra observe      # OpenTelemetry tracing
uv sync --extra all          # Everything above

# Register grip command globally (from source)
uv tool install .
```

### Optional: Multi-Model Support

To use non-Claude models (OpenAI, DeepSeek, Groq, Gemini, Ollama Cloud/Local, etc.), install the LiteLLM extra:

```bash
uv sync --extra litellm
```

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key (for Claude Agent SDK) or an API key from another LLM provider (with `--extra litellm`)

## <a id="quickstart"></a>Quickstart

### 1. Run the Setup Wizard

```bash
grip onboard
```

The wizard walks you through:
- Choosing your engine: **Claude Agent SDK** (recommended) or LiteLLM (15+ providers)
- Entering your API key
- Choosing a default model
- Initializing the workspace (`~/.grip/workspace/`)
- Testing connectivity

### 2. Chat with the Agent

```bash
# Interactive mode
grip agent

# One-shot message
grip agent -m "What files are in my workspace?"

# Pipe input from stdin
cat error.log | grip agent -m "Fix this error"
```

Interactive mode supports slash commands:

| Command | Description |
|---------|-------------|
| `/new` | Start a fresh conversation |
| `/clear` | Clear conversation history |
| `/undo` | Remove last exchange |
| `/rewind N` | Rewind N exchanges |
| `/compact` | Compress session history |
| `/model <name>` | Switch AI model |
| `/status` | Show session info |
| `/doctor` | Run diagnostics |
| `/help` | List all commands |
| `/exit` | Exit grip |

### 3. Start the Gateway

The gateway is the long-running process that connects everything — Telegram, cron, heartbeat, and the REST API:

```bash
grip gateway
```

## <a id="telegram-setup"></a>Telegram Setup

### Step 1: Create a Bot

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the bot token (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### Step 2: Configure grip

```bash
# Set your bot token
grip config set channels.telegram.enabled true
grip config set channels.telegram.token "YOUR_BOT_TOKEN"
```

**Optional:** Restrict the bot to specific Telegram user IDs for security:

```bash
# Find your Telegram user ID by messaging @userinfobot
grip config set channels.telegram.allow_from '["YOUR_TELEGRAM_USER_ID"]'
```

### Step 3: Start the Gateway

```bash
grip gateway
```

You should see:

```
Channels started: telegram
API server started: http://127.0.0.1:18800
grip gateway running. Press Ctrl+C to stop.
```

### Step 4: Chat with Your Bot

Open Telegram and message your bot. Available bot commands:

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | List commands |
| `/new` | Start fresh conversation |
| `/status` | Session info (model, message count, memory) |
| `/model <name>` | Switch AI model (e.g. `/model openai/gpt-4o`) |
| `/trust <path>` | Trust a directory or list trusted dirs (e.g. `/trust ~/Downloads`) |
| `/trust revoke <path>` | Remove a directory from the trusted list |
| `/undo` | Remove last exchange |
| `/clear` | Clear conversation history |
| `/compact` | Summarize and compress session |

Send any text message to chat with the AI. The bot also handles photos (captions), documents, and voice messages.

### Working with Telegram

**Setting Reminders:**

Tell the agent to remind you of something — it creates a cron job and delivers the result back to your Telegram chat:

> "Remind me to check the server status in 30 minutes"

The agent will create a cron job with `--reply-to` pointed at your Telegram chat, so the reminder is delivered directly to you.

**Switching Models:**

```
/model openrouter/google/gemini-2.5-pro
```

**Starting Fresh:**

```
/new
```

**Compressing Long Sessions:**

When a conversation gets long, compress it to save tokens:

```
/compact
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `grip onboard` | Interactive setup wizard |
| `grip agent` | Chat with the AI agent (interactive or one-shot) |
| `grip gateway` | Run full platform: API + channels + cron + heartbeat |
| `grip serve` | Start standalone REST API server |
| `grip status` | Show system status |
| `grip update` | Pull latest source and re-sync dependencies |
| `grip config` | View and modify configuration (`show`, `set`, `path`) |
| `grip cron` | Manage scheduled jobs (`list`, `add`, `remove`, `enable`, `disable`) |
| `grip skills` | Manage agent skills (`list`, `install`, `remove`) |
| `grip workflow` | Manage multi-agent workflows (`list`, `show`, `run`, `create`, `delete`) |
| `grip mcp` | Manage MCP server configurations (`list`, `add`, `remove`, `presets`) |

Global flags: `--verbose` / `-v`, `--quiet` / `-q`, `--config` / `-c PATH`, `--dry-run`.

## <a id="configuration"></a>Configuration

Config is stored at `~/.grip/config.json`. Environment variables override with `GRIP_` prefix and `__` nested delimiter.

```bash
# View current config (secrets are masked)
grip config show

# Set values
grip config set agents.defaults.model "anthropic/claude-sonnet-4"
grip config set agents.defaults.max_tokens 16384
grip config set agents.defaults.temperature 0.7
```

### Key Sections

| Section | Description |
|---------|-------------|
| `agents.defaults` | Engine (`claude_sdk`/`litellm`), SDK model, default model, max_tokens, temperature, memory_window, workspace path |
| `agents.profiles` | Named agent configs (model, tools_allowed, tools_denied, system_prompt_file) |
| `agents.model_tiers` | Cost-aware routing: different models for low/medium/high complexity prompts |
| `providers` | Per-provider API keys and base URLs |
| `tools` | Web search config, shell_timeout, workspace sandboxing, MCP servers |
| `channels` | Telegram/Discord/Slack tokens, allow_from lists |
| `gateway` | Host, port, API auth, rate limits, CORS, request size limits |
| `heartbeat` | Periodic autonomous agent runs (enabled, interval_minutes) |
| `cron` | Scheduled task settings (exec_timeout_minutes) |

### Agent Profiles

Define specialized agents for different tasks:

```json
{
  "agents": {
    "profiles": {
      "researcher": {
        "model": "openai/gpt-4o",
        "tools_allowed": ["web_search", "web_fetch"],
        "system_prompt_file": "RESEARCHER.md"
      },
      "coder": {
        "model": "anthropic/claude-sonnet-4",
        "tools_denied": ["exec"],
        "temperature": 0.3
      }
    }
  }
}
```

### Cost-Aware Model Routing

Route prompts to different models based on complexity:

```bash
grip config set agents.model_tiers.enabled true
grip config set agents.model_tiers.low "openrouter/google/gemini-flash-2.0"
grip config set agents.model_tiers.high "openrouter/anthropic/claude-sonnet-4"
```

Simple queries (greetings, lookups) go to the cheap model. Complex tasks (architecture, debugging) go to the powerful model.

## MCP Servers

grip supports [Model Context Protocol](https://modelcontextprotocol.io) servers via stdio or HTTP/SSE transport.

```bash
# List available presets
grip mcp presets

# Add preset servers
grip mcp presets todoist excalidraw firecrawl

# Add all presets
grip mcp presets --all

# Add custom HTTP server
grip mcp add myserver --url https://mcp.example.com

# Add custom stdio server
grip mcp add myserver --command npx --args -y,mcp-server

# List configured servers
grip mcp list

# Remove a server
grip mcp remove excalidraw
```

### Available Presets

| Name | Type | Description |
|------|------|-------------|
| `todoist` | stdio | Task management |
| `excalidraw` | HTTP | Collaborative whiteboard |
| `firecrawl` | stdio | Web scraping (requires API key) |
| `bluesky` | stdio | Social network |
| `filesystem` | stdio | File system access |
| `git` | stdio | Git operations |
| `memory` | stdio | Knowledge persistence |
| `postgres` | stdio | PostgreSQL queries |
| `sqlite` | stdio | SQLite database |
| `fetch` | stdio | HTTP fetching |
| `puppeteer` | stdio | Browser automation |

Set API keys for presets that require them:

```bash
grip config set tools.mcp_servers.firecrawl.env.FIRECRAWL_API_KEY "your-key"
```

## Workflows

Create multi-agent workflows as JSON files:

```json
{
  "name": "research-and-summarize",
  "description": "Research a topic and produce a summary",
  "steps": [
    {
      "name": "research",
      "prompt": "Research the latest developments in quantum computing",
      "profile": "researcher",
      "timeout_seconds": 600
    },
    {
      "name": "summarize",
      "prompt": "Summarize the following research: {{research.output}}",
      "profile": "coder",
      "depends_on": ["research"]
    }
  ]
}
```

```bash
grip workflow create research.json
grip workflow run research-and-summarize
grip workflow list
```

Steps with no dependencies execute in parallel. The engine uses Kahn's algorithm for topological ordering and detects cycles at validation time.

## Cron & Scheduling

Schedule tasks and reminders with cron expressions or natural language:

```bash
# Add a reminder
grip cron add "standup" "0 9 * * 1-5" "Remind the user: standup in 15 minutes"

# Add a task that reports to Telegram
grip cron add "disk-check" "0 */6 * * *" "Check disk usage" --reply-to "telegram:YOUR_CHAT_ID"

# Manage jobs
grip cron list
grip cron disable <job-id>
grip cron enable <job-id>
grip cron remove <job-id>
```

When chatting via Telegram/Discord/Slack, the agent automatically sets `--reply-to` so cron results are delivered to your chat.

## <a id="api-reference"></a>API Reference

Start the API server standalone or as part of the gateway:

```bash
# Standalone
grip serve

# Full gateway (API + channels + cron + heartbeat)
grip gateway
```

On first run, an auth token is auto-generated and printed to stderr.

### Endpoints

All `/api/v1/*` endpoints require `Authorization: Bearer <token>`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Load balancer probe (no auth) |
| GET | `/api/v1/health` | Authenticated health with version + uptime |
| POST | `/api/v1/chat` | Send message, get response |
| GET | `/api/v1/sessions` | List session keys |
| GET | `/api/v1/sessions/{key}` | Session detail |
| DELETE | `/api/v1/sessions/{key}` | Delete session |
| GET | `/api/v1/tools` | List tool definitions |
| POST | `/api/v1/tools/{name}/execute` | Execute tool directly (disabled by default) |
| GET | `/api/v1/status` | System status |
| GET | `/api/v1/config` | Masked config dump |
| GET | `/api/v1/metrics` | Runtime metrics |
| GET | `/api/v1/cron` | List cron jobs |
| POST | `/api/v1/cron` | Create cron job |
| DELETE | `/api/v1/cron/{id}` | Delete cron job |
| POST | `/api/v1/cron/{id}/enable` | Enable cron job |
| POST | `/api/v1/cron/{id}/disable` | Disable cron job |
| GET | `/api/v1/skills` | List loaded skills |
| GET | `/api/v1/memory` | Read MEMORY.md |
| GET | `/api/v1/memory/search?q=...` | Search HISTORY.md |
| GET | `/api/v1/workflows` | List workflows |
| GET | `/api/v1/workflows/{name}` | Workflow detail |

### Example Requests

```bash
# Health check
curl http://localhost:18800/health

# Chat
curl -X POST http://localhost:18800/api/v1/chat \
  -H "Authorization: Bearer grip_YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?"}'

# List tools
curl -H "Authorization: Bearer grip_YOUR_TOKEN" http://localhost:18800/api/v1/tools

# Metrics
curl -H "Authorization: Bearer grip_YOUR_TOKEN" http://localhost:18800/api/v1/metrics
```

### Security

The API is designed for safe self-hosting:

- **Bearer token auth** with timing-safe comparison
- **Rate limiting** — per-IP (30/min) + per-token (60/min), sliding window
- **Request size limit** — 1 MB default, rejected before parsing
- **Security headers** — X-Content-Type-Options, X-Frame-Options, Content-Security-Policy
- **Audit logging** — every request logged (method, path, status, duration, IP)
- **Directory Trust Model** — grip is restricted to its workspace by default. Access to external directories must be explicitly granted via CLI prompt or the `/trust` command. [Learn more](#security-architecture).
- **Shell Safety Guards** — every shell command is scanned against a comprehensive deny-list (rm -rf /, shutdown, etc.) before execution. [Learn more](#security-architecture).
- **Shield Policy** — context-based runtime threat feed injected into the system prompt. Evaluates tool calls, skill execution, MCP interactions, network requests, and secret access against active threats. [Learn more](#security-architecture).
- **Sanitized errors** — no stack traces or file paths in responses
- **Tool execute gated** — disabled by default to prevent arbitrary command execution over HTTP
- **No config mutation over HTTP** — prevents redirect attacks
- **Startup warnings** — alerts for dangerous configs (0.0.0.0 binding, tool execute enabled)

### Security Architecture

Grip implements a multi-layered defense to make the platform as safe as possible for self-hosting.

#### 1. Directory Trust Model

Grip is sandboxed to its workspace by default. Unlike traditional agents with unrestricted disk access, Grip cannot touch your personal files unless you "opt-in".

- **Workspace First**: The agent can always read/write within its assigned `workspace` directory.
- **Explicit Consent**: To access any directory outside the workspace (e.g., `~/Downloads`), the user must explicitly "trust" it.
- **Persistent Safety**: Trust decisions are saved and remembered across sessions.

#### 2. Shield Policy (Runtime Threat Feed)

The agent's system prompt includes a `SHIELD.md` policy that defines how to evaluate actions against a threat feed before execution. This is a context-level safety layer that works alongside the code-level guards.

**Scopes covered:** `prompt`, `skill.install`, `skill.execute`, `tool.call`, `network.egress`, `secrets.read`, `mcp`

**Enforcement actions:**
- **block** — stop immediately, no execution
- **require_approval** — ask the user for confirmation before proceeding
- **log** — continue normally (default when no threat matches)

**How it works:**
1. Active threats are injected into the `## Active Threats` section of SHIELD.md at runtime
2. Before acting on a scoped event, the agent evaluates it against loaded threats
3. Matching uses category/scope alignment, `recommendation_agent` directives, and exact string fallback
4. Confidence threshold (>= 0.85) determines enforceability; below that, the action defaults to `require_approval`
5. When multiple threats match, the strictest action wins: `block > require_approval > log`

The policy is stored at `workspace/SHIELD.md` and can be customized per workspace.

#### 3. Shell Command Deny-List

Every command the agent tries to run via the `exec` tool is scanned against a robust list of **50+ dangerous patterns** before execution:

- **Destructive Commands**: Blocked `rm -rf /`, `rm -rf ~`, `mkfs`, etc.
- **System Control**: Blocked `shutdown`, `reboot`, `systemctl poweroff`.
- **Credential Exfiltration**: Blocked `cat ~/.ssh/id_rsa`, `cat .env`, etc.
- **Remote Code Injection**: Blocked `curl | bash` and similar pipe-to-shell patterns.

> [!IMPORTANT]
> While we strive for "perfect safety" through these multi-layered guards, no system is infallible. Always run grip with a non-root user and review critical actions.

#### Why it's better:
- **Zero-Trust for sensitive data**: Even top-tier LLMs can "hallucinate". Our guards make it physically impossible for the agent to exfiltrate your SSH keys or delete your home folder by accident.
- **Controlled Blast Radius**: By restricting the agent to specific folders, you ensure an accidental "delete all" command only affects the project directory you're working in.
- **Privacy by Design**: You maintain absolute control over the agent's data footprint.

#### Managing Trust:
- **In Chat (Telegram/Discord)**:
  - `/trust <path>` — Grant permanent access to a directory.
  - `/trust revoke <path>` — Remove access for a directory.
  - `/trust` — List all currently trusted directories.
- **Manual Control**:
  - Trust decisions are stored in `your_workspace/state/trusted_dirs.json`. You can manually edit or clear this file to manage access at any time.

## Docker

Grip is Docker-ready and can be configured entirely via environment variables. The Dockerfile includes Node.js (required by the Claude Agent SDK).

```bash
# Build from source
docker build -t grip .

# Run with Claude Agent SDK (recommended)
docker run -d \
  -p 18800:18800 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e GRIP_CHANNELS__TELEGRAM__ENABLED="true" \
  -e GRIP_CHANNELS__TELEGRAM__TOKEN="bot-token" \
  -v ~/.grip:/root/.grip \
  --name grip-agent \
  grip

# Run with LiteLLM engine (non-Claude models)
docker run -d \
  -p 18800:18800 \
  -e GRIP_AGENTS__DEFAULTS__ENGINE="litellm" \
  -e GRIP_PROVIDERS__OPENAI__API_KEY="sk-..." \
  -e GRIP_CHANNELS__TELEGRAM__ENABLED="true" \
  -e GRIP_CHANNELS__TELEGRAM__TOKEN="bot-token" \
  -v ~/.grip:/root/.grip \
  --name grip-agent \
  grip
```

### Configuration via Environment
Grip supports `GRIP_` prefixed variables for any config value. Use `__` for nested keys:
- `GRIP_AGENTS__DEFAULTS__MODEL`
- `GRIP_PROVIDERS__ANTHROPIC__API_KEY`
- `GRIP_GATEWAY__PORT`

## Built-in Skills

| Skill | Description |
|-------|-------------|
| `cron` | Schedule tasks, reminders, and recurring jobs |
| `github` | PR generation, code review, git workflows |
| `debug` | Bug finding, git blame, bisect, time-travel debugging |
| `code-review` | Automated code review and quality analysis |
| `code-loader` | AST-aware chunking for loading relevant code |
| `codebase-mapper` | Dependency graphs, import mapping, ripple analysis |
| `web-researcher` | Web research with source citation |
| `project-planner` | Project planning and task breakdown |
| `data-viz` | ASCII charts and data visualization in terminal |
| `memory` | Long-term memory management |
| `temporal-memory` | Time-aware reminders and deadline tracking |
| `summarize` | Text and conversation summarization |
| `self-analyzer` | Performance and architecture analysis |
| `skill-creator` | Create new skills |
| `optimization-rules` | Token efficiency and tool selection guidance |
| `yfinance` | Stock market data and analysis |
| `tmux` | Terminal multiplexer management |
| `tweet-writer` | Social media content drafting |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linter
uv run ruff check grip/ tests/

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=grip

# Build package
uv build
```

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run linting and tests: `uv run ruff check grip/ tests/ && uv run pytest`
5. Commit your changes: `git commit -m "Add my feature"`
6. Push to your fork: `git push origin feature/my-feature`
7. Open a Pull Request

Please ensure:
- All tests pass (`uv run pytest`)
- No lint errors (`uv run ruff check grip/ tests/`)
- New features include tests where appropriate

## License

MIT

## Disclaimer

**grip** is an AI-powered platform designed for high autonomy. Please be aware:
- AI models can produce hallucinations, errors, or unexpected outputs.
- Autonomous tool execution (especially shell/exec) carries inherent security risks.
- Users are responsible for monitoring agent behavior and ensuring compliance with LLM provider terms.
- Use this software at your own risk.

