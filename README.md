# Olive Model Optimization

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that helps AI agents query, configure, and troubleshoot Microsoft Olive model optimization workflows.

## Overview

This MCP server provides 32 tools for:

- **Pass Catalog & Configuration** - List 92+ optimization passes, generate config templates, deep-dive into parameters
- **Strategy & Recommendations** - Get quantization strategies, hardware optimization guides, tradeoff analysis
- **Troubleshooting & Diagnostics** - Diagnose Olive errors against the knowledge base with actionable workarounds
- **Compatibility & Validation** - Check model × pass × hardware compatibility and validate UIState configurations
- **Documentation & Reference** - Semantic search across Olive documentation, generate CLI commands, get data configs
- **Job Lifecycle (Studio Integration)** - Submit, monitor, cancel optimization jobs through Olive Studio

## Installation

### Prerequisites

- Python 3.10+ with pip
- Node.js and npm (for MCP integration in Kiro)

### Setup

This repository uses the Agent Plugins format. Install it through a compatible plugin/Powers workflow from the repository root; `plugin.json` is the canonical manifest and the root `mcp.json` contains the MCP server configuration.

For the legacy Kiro Power workflow, `POWER.md` is retained as optional documentation.

Install the MCP server dependencies:

```bash
cd olive-mcp-server
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]" "mcp<2"
```

## Usage

### With Kiro (Recommended)

Install this power in Kiro, then use the tools directly:

```typescript
// Example: Get quantization strategy for LLM on NVIDIA GPU
const result = await get_quantization_strategy({
  model_type: "LLM",
  target_hardware: "NVIDIA RTX 4090",
  latency_budget: "<100ms",
  accuracy_threshold: "<2% drop"
});
```

### Direct CLI Usage

```bash
# Run the MCP server
python olive-mcp-server/run.py

# Or using the console script
.venv\Scripts\olive-mcp-server
```

## Tools

| Tool | Description |
| ------ | ------------- |
| `get_olive_passes` | List available passes, filtered by category |
| `get_pass_config_template` | Generate scaffold Olive workflow JSON for a pass |
| `get_quantization_strategy` | Recommend a quantization approach for model + hardware |
| `get_hardware_optimization_guide` | Return a hardware-specific optimization path |
| `get_pass_chain` | Validate and explain an ordered pass chain |
| `troubleshoot_olive_error` | Diagnose Olive or Studio errors |
| `diagnose_error` | Diagnose an error using the Olive and Studio knowledge bases |
| `get_error_frequency_summary` | Summarize recurring troubleshooting errors |
| `get_model_compatibility` | Check Olive support for a model/framework combo |
| `get_cli_command` | Generate a ready-to-run Olive CLI command |
| `get_data_config_template` | Generate data configuration templates for calibration and evaluation |
| `search_olive_documentation` | Search across the local knowledge base and live docs |
| `get_pass_parameters` | Deep-dive into a pass parameter schema |
| `evaluate_optimization_tradeoff` | Analyze quality vs. performance tradeoff |
| `get_integration_recipe` | Return full Olive recipe templates |
| `get_context_for_pipeline` | Return relevant passive context for a pipeline |
| `validate_ui_state_recipe` | Validate a Studio UIState via local bridge |
| `get_recipe_for_ui_state` | Generate complete Olive recipe from UIState |
| `get_runtime_ep_hints` | Recommend runtime execution-provider settings |
| `record_troubleshoot_feedback` | Local aggregate thumbs feedback for KB entries |
| `get_mcp_capabilities` | Capability state for agents |
| `list_optimization_jobs` | List Studio jobs |
| `get_optimization_job` | Get status for one Studio job |
| `get_optimization_results` | Get results/metrics from a completed job |
| `validate_optimization_job` | Studio preflight + fingerprint |
| `submit_optimization_job` | Submit an optimization job via Studio |
| `cancel_optimization_job` | Cancel a running job |
| `plan_optimization` | Create a staged optimization plan |
| `execute_and_observe` | Execute and observe an optimization plan |
| `diagnose_and_fix` | Diagnose failed runs and propose fixes |
| `compare_results` | Compare optimization results |
| `get_model_info` | Inspect model metadata |

## Configuration

### Environment Variables

| Variable | Purpose |
| ---------- | --------- |
| `OLIVE_MCP_RETRIEVAL_MODE` | `auto` (default), `keyword`, or `semantic` |
| `OLIVE_MCP_SEMANTIC_BUDGET_MS` | Cold semantic budget (default 8000; 0 = unlimited) |
| `OLIVE_MCP_PRELOAD_EMBEDDINGS` | If `1`, warm model + indexes at startup |
| `OLIVE_STUDIO_API_URL` | Loopback base URL for Studio (e.g., `http://127.0.0.1:3000`) |

### MCP Configuration

The canonical Agent Plugins configuration is the root [`mcp.json`](./mcp.json). It uses `${PLUGIN_ROOT}` so the installed plugin can locate the unchanged internal `olive-mcp-server` directory on any machine.

```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
  "mcpServers": {
    "olive-mcp": {
      "type": "stdio",
      "command": "python",
      "args": ["run.py"],
      "cwd": "${PLUGIN_ROOT}/olive-mcp-server",
      "env": {
        "OLIVE_MCP_RETRIEVAL_MODE": "auto",
        "PYTHONPATH": "${PLUGIN_ROOT}/olive-mcp-server"
      }
    }
  }
}
```

## Schema and Knowledge-Base Provenance

The authoritative recipe reference is the [Microsoft Olive schema](https://microsoft.github.io/Olive/schema.json). Generated pass templates and detailed integration recipes use the current Olive 0.13 conventions: `input_model` is flat (`input_model.model_path`, not `input_model.config.model_path`), Hugging Face model IDs use `HfModel`, pass entries are arrays of `{ "type": ..., "config": ... }`, and data components use `{ "type": ..., "params": ... }`.

The bundled `passes.json`, compatibility matrix, and raw `integration_recipes.json` are intentionally retained as **historical 0.13.0 reference snapshots** for compatibility guidance and troubleshooting. They are not emitted verbatim: `get_integration_recipe` migrates detailed recipes to the current flat model, pass, and data-component shapes before returning them. They are not proof that every listed pass is available in the latest Olive release. Use the `schema_source`, `catalog_status`, and evidence fields in tool responses to distinguish current recipe output from historical catalog data.

## Knowledge Base

The server includes a local knowledge base with:

- `passes.json` - Pass catalog with parameters and gotchas
- `hardware_profiles.json` - Hardware target profiles
- `quirks.json` - Common behaviors and pitfalls
- `troubleshooting.json` - Olive runtime error diagnosis rules
- `studio_troubleshooting.json` - Olive Studio / builder / UI diagnosis rules
- `compatibility_matrix.json` - Model compatibility matrix (evidence-backed claims)
- `integration_recipes.json` - Ready-to-run Olive recipe templates

## Development

### Running Tests

```bash
cd olive-mcp-server
.venv\Scripts\python -m pytest tests -q
```

### Updating the Knowledge Base

```bash
.venv\Scripts\python -m scripts.update_kb
.venv\Scripts\python -m scripts.expand_kb
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests -q`
4. Submit a pull request

## Support

For bugs, schema mismatches, or MCP connection issues, open a [GitHub support issue](https://github.com/tonythethompson/olive-model-optimization/issues). Please include the tool name, a minimal sanitized input, your Python version, and the MCP server logs when possible.

## Privacy Policy

Read the repository [Privacy Policy](./PRIVACY.md).

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Knowledge base built using [sentence-transformers](https://huggingface.co/sentence-transformers)
- Olive documentation scraped from <https://microsoft.github.io/Olive/>
