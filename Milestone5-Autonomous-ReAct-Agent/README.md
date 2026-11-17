# Milestone 5 – Autonomous ReAct Agent

## Overview
A local ReAct-based AI agent powered by Gemma2:2b through Ollama.

The agent uses a text-based ReAct protocol to decide which tool to execute, observe the result, and continue until it produces a final answer or reaches the maximum iteration limit.

## Available Tools
1. **SQLite Query:** Executes read-only SELECT queries on a local SQLite database.
2. **Calculator:** Performs safe mathematical calculations using Python AST parsing.
3. **File Search:** Searches filenames and file contents inside a local workspace.

## Requirements
- Python 3
- Ollama
- Gemma2:2b
- requests

Install the Python dependency:

```bash
python -m pip install requests
```

Make sure Ollama is running and the model is available:

```bash
ollama pull gemma2:2b
```

## Run

```bash
python milestone5_agent.py
```

Ask questions interactively, or run:

```bash
python milestone5_agent.py test all
python milestone5_agent.py test 6
python milestone5_agent.py safety
```

## Testing
- Tests 1–5: Agent tool usage and multi-step tasks.
- Test 6: Deterministic iteration-limit test.
- Safety checks: Validation of restricted SQL, calculator inputs, and unknown tools.

## Notes
The sample SQLite database and workspace files are created automatically on the first run.

The project uses a text-based ReAct protocol because the selected local model does not support native Ollama tool calling.