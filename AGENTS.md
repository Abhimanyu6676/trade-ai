# trade-ai Agent Instructions

Last reviewed: 2026-11-01.

## DO

### Before working

- Read this file first, then read `context.md` for project architecture, data/model contracts, runtime behavior, and known gotchas. After that, inspect only the source files relevant to the task.
- Use `readme.md` for the user-facing project summary and structure, and `setup-start.md` for installation and startup procedures.
- Before refactoring training code, read `src/trade_ai/model/training/training.md` and any other relevant training notes.
- Before refactoring feature code, read `src/trade_ai/model/features.md` and any other relevant feature notes.
- Treat `src/trade_ai/model/features.md` as the source of truth for feature rules, scaling notes, per-model input counts, and complete feature lists.

### Code and refactoring

- Prefer object-oriented design where it fits the module, and follow Don't Repeat Yourself (DRY) principles by reusing shared logic instead of duplicating it.
- When creating or refactoring functions, add a brief docstring describing the function's purpose and detailed `#` comments explaining its workflow and the specific job of each meaningful code block.
- Keep types with the module they describe in that module's `*Types.py` file. Routes should import and reuse those types when defining request or response contracts; for example, OHLCV candle data and validation belong in `openalgo/openalgoTypes.py`.
- When refactoring server/API code or changing anything that may affect frontend integration—including endpoint behavior, request parameters or body, or response values or shape—ask the user whether they want corresponding changes made in the sibling `trade-ai-chat` project. Wait for their answer before making those frontend changes.

### Documentation and maintenance

- Keep project-specific facts and technical behavior in `context.md`; keep this file limited to agent workflow and repository maintenance instructions.
- When architecture, runtime behavior, data contracts, startup, environment requirements, or operational gotchas change, update the relevant sections of `context.md` in the same change. Feature count, name, timeframe, or scaling changes alone belong in `features.md`; update `context.md` for feature-related structural/context changes such as renaming or moving the feature module.
- Whenever features or feature-building behavior change, update `src/trade_ai/model/features.md` in the same change with the total input count and full feature-name/style list for both models, even if only one model's schema changed.
- For every added feature, assess its normalization and whether it is relative/scale-independent. Tell the user about any feature that is unnormalized or non-relative, and distinguish bounded, binary, or one-hot features from raw scale-dependent values.
- When the user prompts `check/evaluate features`, perform and report a feature audit for every model. For each model, include the total number of actual model input features and list every input feature name with its style in brackets **before** the name (for example, `[relative] ROC_1m`, `[percentage %] Return_1m`, `[one-hot flag 0/1] Cap_Large`, or `[RSI 0-1] RSI_14_1m`). Base counts and lists on the model-specific input schemas; exclude raw OHLCV columns, timestamps, labels, and intermediate calculations. Explicitly flag any unnormalized or non-relative input, including bounded/binary inputs while noting their bounded or categorical nature.
- Update `setup-start.md` when setup or startup instructions change.
- Update `readme.md` when the project summary, major capabilities, or structure changes.
- Treat `src/trade_ai/server/server.md` as the server module and API endpoint guidebook and rulebook. Read and follow it before changing server code or endpoints, and update it whenever an endpoint or server behavior changes.
- When training logic or lifecycle steps change, update `src/trade_ai/model/training/training.md` in the same change.

## DON'T

- Never include raw OHLCV values as model input features. Keep raw OHLCV in the source frame for calculations only.
- Never write credential values, tokens, or private environment values into tracked files or logs.
- Do not add or run tests unless the user asks. No test command is currently configured in `pyproject.toml`.
- Don't touch `codex-accounts.json`