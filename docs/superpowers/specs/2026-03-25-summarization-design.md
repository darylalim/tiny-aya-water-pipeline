# Summarization Feature Design

## Overview

Add cross-lingual summarization to the Tiny Aya Water app. Users can input text and receive a summary in any of the 43 supported languages, with control over summary length (short/medium/long). Supports both single text and batch file summarization.

## Approach

Extend the existing single-file architecture (`streamlit_app.py`). No new files, dependencies, or environment variables.

## Modifications to Existing Code

### Rename `extract_translation` -> `clean_model_output`

The existing `extract_translation(text: str) -> str` function is renamed to `clean_model_output` for task-neutral naming. Same implementation (`.strip()`). All call sites are updated:
- `translate_text` in `streamlit_app.py` (currently calls `extract_translation(decoded)`)
- Test imports in `test_streamlit_app.py` (line 11: `extract_translation` -> `clean_model_output`)
- Four existing test functions renamed: `test_extract_translation_*` -> `test_clean_model_output_*`

### Update page title and description

- `st.title` changes from "Tiny Aya Water Translator" to "Tiny Aya Water" (task-neutral)
- `st.markdown` description updated to mention both translation and summarization

## New Pure Functions

Three new functions above `import streamlit`, following the existing pattern:

### `get_summary_config(length: str) -> str`

Maps a summary length label to a prompt instruction string. Raises `ValueError` for unrecognized length values.

| Length | Prompt instruction                          |
|--------|---------------------------------------------|
| Short  | "Write a brief summary in 1-2 sentences"    |
| Medium | "Write a summary in a short paragraph"      |
| Long   | "Write a detailed summary"                  |

### `build_summarization_prompt(text: str, summary_length: str, target_lang: str) -> list[dict[str, str]]`

Constructs a chat message list for summarization. Calls `get_summary_config` to get the length instruction. Returns a message list containing a single user message asking the model to summarize the text in the target language at the specified length. Output format matches `build_translation_prompt`.

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

No changes. Temperature, max tokens slider, and model info remain shared across both tasks. The max tokens slider always defaults to `DEFAULT_MAX_TOKENS` regardless of task or summary length. Summary length controls the model's output through prompt wording only — the slider serves as a hard token cap the user can adjust manually. The `get_summary_config` default max tokens values are not wired to the slider.

## Error Handling

- **Empty input**: show `st.warning("Please enter some text to summarize.")` and do not call the model, matching the translation flow.
- **Batch generation failures**: if `summarize_text` raises during a batch row, catch the exception, show `st.warning` for the failed row, insert `"[Error: generation failed]"` in the result for that row, and continue processing remaining rows. This is new behavior for summarization only — the translation batch flow remains unchanged (no per-row error handling).
- **Short input with long summary mode**: no special handling — the model naturally handles this. The prompt instructs the desired length; the model will produce what it can.

## Config and Environment

No new environment variables. Summary length defaults are hardcoded in `get_summary_config` since they are UI presets, not deployment config. Existing config (`DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`, `MAX_BATCH_ROWS`) applies to summarization as-is.

## Tests

New tests in `test_streamlit_app.py`, following existing patterns. Existing `extract_translation` tests are renamed to `clean_model_output` (same assertions, covered in "Modifications to Existing Code" above).

### `get_summary_config` (~4 tests)

- Returns correct prompt instruction for each length (short, medium, long).
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
