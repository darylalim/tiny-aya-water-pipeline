# Summarization Feature Design

## Overview

Add cross-lingual summarization to the Tiny Aya Water app. Users can input text and receive a summary in any of the 43 supported languages, with control over summary length (short/medium/long). Supports both single text and batch file summarization.

## Approach

Extend the existing single-file architecture (`streamlit_app.py`). No new files, dependencies, or environment variables.

## New Pure Functions

Three new functions above `import streamlit`, following the existing pattern:

### `get_summary_config(length: str) -> tuple[str, int]`

Maps a summary length label to a prompt instruction and default max tokens. Raises `ValueError` for unrecognized length values.

| Length | Prompt instruction                          | Default max tokens |
|--------|---------------------------------------------|--------------------|
| Short  | "Write a brief summary in 1-2 sentences"    | 150                |
| Medium | "Write a summary in a short paragraph"      | 350                |
| Long   | "Write a detailed summary"                  | 700                |

### `build_summarization_prompt(text: str, summary_length: str, target_lang: str) -> list[dict[str, str]]`

Constructs a chat message list for summarization. Calls `get_summary_config` to get the length instruction. Returns a single user message asking the model to summarize the text in the target language at the specified length. Output format matches `build_translation_prompt`.

### `clean_model_output(text: str) -> str`

Rename of existing `extract_translation` to a task-neutral name. Same implementation (`.strip()`). Both `translate_text` and `summarize_text` will call this function. Existing tests for `extract_translation` are updated to reference the new name.

### `summarize_text(text: str, target_lang: str, summary_length: str, model: Any, tokenizer: Any, temperature: float, max_tokens: int) -> str`

End-to-end summarization: builds prompt, tokenizes, generates, decodes. Mirrors `translate_text` internally — same `apply_chat_template` / `model.generate` / `tokenizer.decode` flow with the same BatchEncoding handling. Uses `clean_model_output` for output cleanup.

## UI Changes

### Task Selector

A horizontal `st.radio("Task", ["Translate", "Summarize"])` placed below the title and description. The selected task determines which UI section is shown.

### Translate Mode

Identical to the current UI. No changes.

### Summarize Mode

- **Summary length**: `st.radio("Summary Length", ["Short", "Medium", "Long"], horizontal=True)`
- **Output language**: `st.selectbox("Output Language", LANGUAGES)` — the language the summary is written in. No source language selector (model infers input language).
- **Text area**: for input text.
- **Summarize button**: calls `summarize_text`, displays result in a disabled text area.
- **Batch section**: file uploader (CSV/TXT), reuses `parse_uploaded_file`, iterates rows through `summarize_text` with progress bar, displays results table with columns `{"original": ..., "summary": ...}`, CSV download button (filename: `summaries.csv`).

### Sidebar

Temperature and model info remain shared across both tasks. The max tokens slider is shared but its default value updates based on context: in Summarize mode, switching summary length updates the slider default via `get_summary_config`. The user can still manually adjust the slider to override the length-based default.

## Error Handling

- **Empty input**: show `st.warning("Please enter some text to summarize.")` and do not call the model, matching the translation flow.
- **Batch generation failures**: if `summarize_text` raises during a batch row, catch the exception, insert an error message (e.g., `"[Error: generation failed]"`) for that row, and continue processing remaining rows.
- **Short input with long summary mode**: no special handling — the model naturally handles this. The prompt instructs the desired length; the model will produce what it can.

## Config and Environment

No new environment variables. Summary length defaults are hardcoded in `get_summary_config` since they are UI presets, not deployment config. Existing config (`DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`, `MAX_BATCH_ROWS`) applies to summarization as-is.

## Tests

New tests in `test_streamlit_app.py`, following existing patterns. No changes to existing tests.

### `clean_model_output` (~4 tests, renamed from `extract_translation`)

- Existing tests updated to use the new name. Same assertions.

### `get_summary_config` (~4 tests)

- Returns correct prompt instruction and max tokens for each length (short, medium, long).
- Raises `ValueError` for invalid length.

### `build_summarization_prompt` (~5 tests)

- Returns a single user message.
- Contains the target language.
- Contains the input text.
- Includes a summarization instruction.
- Includes the length-specific wording.

### `summarize_text` (~4 tests, mocked model/tokenizer)

- Plain tensor path: input moved to model device, attention mask is None.
- BatchEncoding path: input_ids and attention_mask moved to model device.
- Returns cleaned string output.
- Passes correct generate parameters (temperature, max_tokens, do_sample).

## Documentation Updates

- Update `pyproject.toml` description to mention summarization.
- Update `README.md` features list to include summarization.
- Update `CLAUDE.md` project description.

## Scope Boundaries

- No new files or modules.
- No new dependencies.
- No new environment variables.
- One rename: `extract_translation` -> `clean_model_output` (used by both tasks). Existing tests updated accordingly.
- No multi-page Streamlit app — stays single-page with task selector.
