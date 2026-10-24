# trade-ai Agent Instructions

Last reviewed: 2026-10-24.

## Before Working

- Read this file first, then read `context.md` for project architecture, data/model contracts, runtime behavior, and known gotchas. After that, inspect only the source files relevant to the task.
- Use `readme.md` for the user-facing project summary and structure, and `setup-start.md` for installation and startup procedures.

## Maintenance

- Keep project-specific facts and technical behavior in `context.md`, not in this instruction file.
- When behavior, architecture, model inputs, startup, environment requirements, or operational gotchas change, update the relevant sections of `context.md` in the same change.
- Update `setup-start.md` when setup or startup instructions change. 
- Update `readme.md` when the project summary, major capabilities, or structure changes.
- Update `server.md` whenever an endpoint is added or modified
- Keep types with the module they describe in that module's `*Types.py` file. Routes should import and reuse those types when defining request or response contracts; for example, OHLCV candle data and validation belong in `openalgo/openalgoTypes.py`.
- Keep this file limited to agent workflow and repository maintenance instructions.
- Never write credential values, tokens, or private environment values into tracked files or logs.
- Do not add or run tests unless the user asks. No test command is currently configured in `pyproject.toml`.
