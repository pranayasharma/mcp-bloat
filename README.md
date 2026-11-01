# mcp-bloat

[![PyPI version](https://img.shields.io/pypi/v/mcp-bloat)](https://pypi.org/project/mcp-bloat/)
[![Python versions](https://img.shields.io/pypi/pyversions/mcp-bloat)](https://pypi.org/project/mcp-bloat/)
[![License: MIT](https://img.shields.io/pypi/l/mcp-bloat)](https://github.com/pranayasharma/mcp-bloat/blob/main/LICENSE)

**Audit how many tokens your MCP servers eat from your context window — before your agent pays for them.**

Every MCP server you connect makes your AI client load that server's tool
schemas into the model's context — on every session, before you've asked a
single question. A handful of servers can quietly consume a noticeable
share of the context window. `mcp-bloat` measures it: it finds your MCP
configs, connects to each server for real, and prints the token bill.

## Install

```bash
pip install mcp-bloat
# or, without installing:
uvx mcp-bloat
```

For exact token counts (tiktoken's `cl100k_base` instead of a chars/4
estimate), install the `accurate` extra:

```bash
pip install "mcp-bloat[accurate]"
```

## Usage

```bash
# Audit every MCP server in your known client configs
mcp-bloat

# See it work without configuring anything: audits three generated demo servers
mcp-bloat --demo

# Machine-readable output
mcp-bloat --json

# Fail CI if your servers eat more than 30,000 tokens of context
mcp-bloat --budget 30000

# Audit one specific config file
mcp-bloat --config ./mcp.json

# Audit one remote server directly, by URL — for example a server you
# use through ChatGPT or another hosted client
mcp-bloat --url https://example.com/mcp

# If that server expects an Authorization header, pass it through
# (--header is repeatable, and --name sets the label shown in the report)
mcp-bloat --url https://example.com/mcp --header "Authorization: Bearer <token>"

# Compute the percentages against a specific model's context window —
# looked up live from OpenRouter/LiteLLM (free, no API key; cached 24h)
mcp-bloat --model gpt-5
mcp-bloat --model claude-sonnet-4-5

# …or set the window size yourself (this overrides --model)
mcp-bloat --window 128000
```

### Which window am I using?

Every percentage in the report is relative to a context window. By
default that's 200,000 tokens. Pass `--model NAME` to measure against
the window of the model you actually use — `mcp-bloat` resolves the name
against live model data from OpenRouter (with LiteLLM as a fallback and
a small built-in snapshot when you're offline), and the report states
which model and window it used. `--window N` skips the lookup and uses
exactly N tokens. A server that eats 14% of a 200k window eats about 22%
of a 128k one — same tokens, different pain.

## Sample output

From `mcp-bloat --demo` (three generated servers: a tidy one, a medium
one, and a deliberately huge one):

```
Demo — mcp-bloat --demo (generated config)
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Server      ┃ Transport ┃ Tools ┃ Tokens ┃ % of 200k ctx ┃ Status  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ notes-lite  │ stdio     │     3 │    204 │          0.1% │ ok      │
│ dev-toolbox │ stdio     │    15 │  2,394 │          1.2% │ ok      │
│ mega-api    │ stdio     │    50 │ 25,164 │         12.6% │ ⚠ BLOAT │
└─────────────┴───────────┴───────┴────────┴───────────────┴─────────┘

╭─ mcp-bloat summary ─────────────────────────────────────────────╮
│ Total: 27,762 tokens  (13.9% of a 200,000-token context window) │
│ Of your 27,762 tokens, that's:                                  │
│   2.8% of Claude Sonnet 5.5's 1,000,000 window                  │
│   6.9% of GPT-5's 400,000 window                                │
│   2.6% of Gemini 2.5 Pro's 1,048,576 window                     │
│   16.9% of DeepSeek V3's 163,840 window                         │
│ Servers audited: 3                                              │
│ Counting method: tiktoken cl100k_base                           │
│ ⚠ Over 10,000 tokens on its own: mega-api (25,164)              │
╰─────────────────────────────────────────────────────────────────╯
```

In default mode (no `--model`/`--window`), the summary also lines your
total up against several current frontier models' context windows at
once — resolved from the same live model data — so you can see the cost
on the models people actually use; pass `--model` or `--window` and the
summary sticks to that single window instead.

Any single server over 10,000 tokens is flagged `⚠ BLOAT`.

## How the counting works

For each configured server, `mcp-bloat`:

1. Connects to the server over MCP (stdio or streamable HTTP) and performs
   the standard initialize handshake.
2. Calls `tools/list` (and `prompts/list` / `resources/list` when the
   server supports them — prompts and resources are surfaced to the model
   too).
3. For every tool, serializes `{name, description, inputSchema}` to JSON —
   the same information a client sends to the model so it can call the
   tool — and counts the tokens in that string.

Token counting uses tiktoken's `cl100k_base` encoding when the
`accurate` extra is installed, and falls back to a chars/4 estimate
otherwise. The report always states which method produced the numbers.

This is a close proxy for what clients actually send, not a byte-exact
replica: individual clients wrap tool definitions slightly differently,
and models with different tokenizers will count a little differently.
Treat the numbers as accurate to a few percent — more than enough to find
the server that's eating your context.

Servers that fail to start or don't respond within the timeout
(default 20s, `--timeout` to change) are reported as errors; they never
abort the rest of the audit. Server stderr is discarded so broken
servers can't corrupt the report.

## Config locations searched

- **Claude Code** — `~/.claude.json`, `./.mcp.json`
- **Cursor** — `~/.cursor/mcp.json`, `./.cursor/mcp.json`
- **Claude Desktop** — `~/.config/Claude/claude_desktop_config.json`,
  `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windsurf** — `~/.codeium/windsurf/mcp_config.json`
- **VS Code** — `./.vscode/mcp.json` (the `servers` key)
- **Gemini CLI** — `~/.gemini/settings.json`, `./.gemini/settings.json`
  (both the `url` and `httpUrl` spellings for HTTP servers)
- **Codex CLI** — `~/.codex/config.toml` (the `[mcp_servers.*]` tables)

Clients that don't keep MCP servers in a local config file — ChatGPT,
for instance — can still be covered: take the server's URL and pass it
to `--url`, with `--header` for any authorization header the server
expects.

Malformed config files produce a warning, not a crash.

## License

MIT
