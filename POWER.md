---
name: "olive-mcp-tools"
displayName: "Olive MCP Tools"
description: "Query the Olive pass catalog, validate recipes, troubleshoot optimization errors, and inspect hardware profiles during development via the project's MCP server."
keywords: ["olive", "onnx", "mcp", "model-optimization", "quantization", "recipe", "troubleshooting", "hardware", "passes"]
author: "Trackdub Team"
---

# Olive MCP Tools Power

This power connects Kiro to the Olive Studio MCP server — a Python FastMCP server with 32 tools covering pass catalog queries, recipe validation, strategy advice, troubleshooting, documentation search, and job lifecycle management.

Use these tools during development to validate recipe configurations, check pass compatibility, troubleshoot optimization errors, and query the knowledge base without leaving the editor.

## Overview

The Olive MCP server provides:

- **Pass Catalog & Configuration** - List 92+ optimization passes, generate config templates, deep-dive into parameters
- **Strategy & Recommendations** - Get quantization strategies, hardware optimization guides, tradeoff analysis
- **Troubleshooting & Diagnostics** - Diagnose Olive errors against the knowledge base with actionable workarounds
- **Compatibility & Validation** - Check model × pass × hardware compatibility and validate UIState configurations
- **Documentation & Reference** - Semantic search across Olive documentation, generate CLI commands, get data configs
- **Job Lifecycle (Studio Integration)** - Submit, monitor, cancel optimization jobs through Olive Studio

## Prerequisites

The MCP server requires a Python virtual environment with dependencies installed:

```bash
cd olive-mcp-server
python -m venv .venv
# Windows:
.venv\Scripts\pip install -e ".[dev]" "mcp<2"
# Linux/macOS:
.venv/bin/pip install -e ".[dev]" "mcp<2"
```

The `mcp` package MUST be pinned `<2` — version 2.x breaks imports.

## Available Tools

### Pass Catalog & Configuration

| Tool | Description |
| ------ | ------------- |
| `get_olive_passes` | List all 92+ optimization passes with optional filtering by type, hardware, format |
| `get_pass_config_template` | Generate a pass configuration JSON template for a specific pass type |
| `get_pass_parameters` | Detailed parameter documentation for a pass (types, defaults, constraints) |
| `get_pass_chain` | Suggest an ordered pipeline of passes for a given optimization goal |

### Strategy & Recommendations

| Tool | Description |
| ------ | ------------- |
| `get_quantization_strategy` | Recommend quantization approach given model, hardware, and accuracy needs |
| `get_hardware_optimization_guide` | Hardware-specific optimization guidance for an execution provider |
| `evaluate_optimization_tradeoff` | Compare tradeoffs between different optimization approaches |
| `get_context_for_pipeline` | Passive context to inform pipeline decisions |

### Troubleshooting & Diagnostics

| Tool | Description |
| ------ | ------------- |
| `troubleshoot_olive_error` | Diagnose an Olive error against the knowledge base |
| `diagnose_error` | Generic error diagnosis (broader than Olive-specific) |
| `get_error_frequency_summary` | Error pattern tracking and frequency stats |
| `record_troubleshoot_feedback` | Submit thumbs-up/down feedback on a diagnosis (improves KB quality) |

### Compatibility & Validation

| Tool | Description |
| ------ | ------------- |
| `get_model_compatibility` | Check model × pass × hardware compatibility |
| `validate_ui_state_recipe` | Validate a Studio UIState JSON against recipe rules |
| `get_recipe_for_ui_state` | Generate complete Olive recipe JSON from a UIState |
| `get_runtime_ep_hints` | Runtime execution provider hints and diagnostics |

### Documentation & Reference

| Tool | Description |
| ------ | ------------- |
| `search_olive_documentation` | Semantic search across Olive documentation (uses embeddings) |
| `get_cli_command` | Generate Olive CLI commands for a given workflow |
| `get_data_config_template` | Data configuration templates for calibration/evaluation |
| `get_integration_recipe` | Pre-built end-to-end integration recipes |
| `get_mcp_capabilities` | Server self-description (tool list, version, features) |

### Job Lifecycle (Studio Integration)

| Tool | Description |
| ------ | ------------- |
| `list_optimization_jobs` | List active and past optimization jobs |
| `get_optimization_job` | Get details for a specific job |
| `get_optimization_results` | Get results/metrics from a completed job |
| `validate_optimization_job` | Pre-validate a job configuration |
| `submit_optimization_job` | Submit a new optimization job |
| `cancel_optimization_job` | Cancel a running job |

## Usage Guidance

### When to use during development

- **Implementing a new pass**: Call `get_olive_passes` to check if the pass exists, `get_pass_parameters` for field details, `get_pass_config_template` for a starter config.
- **Debugging a recipe failure**: Call `troubleshoot_olive_error` with the error text.
- **Checking hardware support**: Call `get_model_compatibility` with model type + target EP.
- **Validating a UIState change**: Call `validate_ui_state_recipe` with the current state JSON.
- **Finding documentation**: Call `search_olive_documentation` with a natural language query.

### Performance Notes

- The MCP server spawns as a subprocess per batch call (~500ms startup). For multiple queries, batch them.
- A circuit breaker protects against repeated failures: 3 consecutive infra failures → 30s cooldown → half-open probe.
- Tool-level errors (bad arguments, no match) do NOT trip the breaker — only infra failures (process crash, timeout, non-JSON output).
- Timeout per call: 45 seconds. Max output buffer: 8MB.

### Semantic Search

`search_olive_documentation` uses sentence-transformers embeddings. First call may take longer (model load). Subsequent calls are fast (cached). Set `OLIVE_MCP_PRELOAD_EMBEDDINGS=1` to warm the model at server start.

## Best Practices

- Prefer `get_olive_passes` with filters over full catalog dumps when you know what you're looking for.
- Use `validate_ui_state_recipe` to catch issues before they reach the UI validation layer.
- When troubleshooting, include the full error traceback — the KB matches on stack patterns.
- Feedback via `record_troubleshoot_feedback` improves diagnosis quality over time.
- The `get_context_for_pipeline` tool is designed for AI assistants to gather relevant context before making recommendations.

## Troubleshooting

### Common Issues

**Error: "No project .venv found"**

- Create the virtual environment: `python -m venv olive-mcp-server/.venv && pip install -e ".[dev]" "mcp<2"`

**Error: "OLIVE_STUDIO_API_URL is not set"**

- Start Olive Studio (`pnpm dev` or `pnpm start`)
- Set `OLIVE_STUDIO_API_URL` to loopback: `http://127.0.0.1:3000`

**Semantic search is slow on first call**

- This is expected — the embedding model loads at runtime
- Set `OLIVE_MCP_PRELOAD_EMBEDDINGS=1` to load at server start

### Environment Variables

| Variable | Purpose |
| ---------- | --------- |
| `OLIVE_MCP_RETRIEVAL_MODE` | `auto` (default), `keyword`, or `semantic` |
| `OLIVE_MCP_SEMANTIC_BUDGET_MS` | Cold semantic budget (default 8000; 0 = unlimited) |
| `OLIVE_MCP_PRELOAD_EMBEDDINGS` | If `1`, warm model + indexes at startup |
| `OLIVE_MCP_REQUIRE_VENV` | If `1`, launcher exits when no project venv is found |
| `OLIVE_STUDIO_API_URL` | Loopback base URL for Studio (e.g., `http://127.0.0.1:3000`) |

## Configuration

The MCP server is configured in `~/.kiro/powers.mcp.json`:

```json
{
  "mcpServers": {
    "power-olive-mcp-tools-olive-mcp": {
      "type": "stdio",
      "command": "python",
      "args": ["run.py"],
      "cwd": "D:\\Dev\\Olive-Studio\\olive-mcp-server",
      "disabled": false,
      "disabledTools": []
    }
  }
}
```

## MCP Config Placeholders

**IMPORTANT:** Before using this power, replace the following placeholders in `mcp.json` with your actual values:

- **`d:/Dev/olive-mcp-powers/olive-mcp-server`**: The absolute path to your olive-mcp-server directory.
  - **How to set it:**
    1. Locate your olive-mcp-powers workspace directory
    2. The server is located at `olive-mcp-server` subdirectory
    3. Use the full absolute path (e.g., `C:/Users/yourname/projects/olive-mcp-powers/olive-mcp-server`)

**After replacing placeholders, your mcp.json should look like:**

```json
{
  "mcpServers": {
    "olive-mcp": {
      "type": "stdio",
      "command": "python",
      "args": ["run.py"],
      "cwd": "C:/Users/yourname/projects/olive-mcp-powers/olive-mcp-server",
      "env": {
        "OLIVE_MCP_RETRIEVAL_MODE": "auto",
        "PYTHONPATH": "C:/Users/yourname/projects/olive-mcp-powers/olive-mcp-server"
      },
      "disabled": false,
      "disabledTools": []
    }
  }
}
```

**Note:** For local development, the current configuration uses `d:/Dev/olive-mcp-powers/olive-mcp-server` which works for the development environment. When sharing this power, users should replace with their actual path.

---

**Package:** `olive-mcp-server`
**MCP Server:** `olive-mcp`
