# Tiny Aya Water Translator — Design Spec

## Overview

A single-page Streamlit application that provides translation between 40+ European and Asia-Pacific languages using the CohereLabs/tiny-aya-water model via local HuggingFace Transformers inference. Supports both single text translation and batch file translation (CSV/TXT) with download.

**Note:** The tiny-aya-water model is licensed CC-BY-NC (non-commercial). A license notice is displayed in the app sidebar.

## Architecture

Single-file Streamlit app (`streamlit_app.py`) with three logical sections:

1. **Model loading** — load `CohereLabs/tiny-aya-water` tokenizer + model once via `@st.cache_resource`, BF16 precision on GPU or FP32 fallback on CPU. Device defaults to `cpu`, configurable via `.env`. Model load wrapped in try/except with `st.error()` on failure and `st.spinner()` during download.
2. **Translation function** — takes source text, source language, target language; constructs a chat-template prompt; runs `model.generate()` and decodes the output
3. **UI layer** — language selectors, text input, file upload, translate button, results display, download

Config (model ID, default generation params, device) loaded from `.env` via `python-dotenv`.

## UI Layout

Top-down single-page flow:

- **Title:** "Tiny Aya Water Translator" with subtitle describing the model
- **Language selectors:** Two columns — source language (default: English) and target language (default: French). Both contain the full list of Water's supported languages as human-readable names
- **Text input:** `st.text_area` for entering text to translate
- **Translate button:** Triggers single-text translation
- **Translation output:** `st.text_area`, read-only, displays the result
- **Batch Translation section:**
  - File upload accepting CSV and TXT
  - Column selector (if CSV) to pick which column to translate
  - Translate File button
  - Results table (`st.dataframe`) showing original + translated columns side by side
  - Download CSV button
- **Sidebar:**
  - Temperature slider (0.0–1.0, default 0.1)
  - Max tokens slider (100–2000, default 700)

## Translation Logic

### Prompt Construction

Uses a single user message combining the instruction and the text. The model's chat template may not support a system role, so we avoid it.

```python
messages = [
    {
        "role": "user",
        "content": "Translate the following text from {source_language} to {target_language}. Output only the translation, nothing else.\n\n{text_to_translate}"
    }
]
```

Applied via `tokenizer.apply_chat_template()`, then `model.generate()`.

### Generation Parameters

- `temperature`: 0.1 (default, configurable via sidebar)
- `max_new_tokens`: 700 (default, configurable via sidebar)
- `do_sample`: True
- `top_p`: 0.95

### Validation

- If source and target language are the same, show `st.warning()` and do not translate
- If text input is empty, show `st.warning()` and do not translate

### Single Text Flow

1. Build messages, tokenize with chat template, generate, decode with `skip_special_tokens=True`
2. Strip any remaining whitespace from the output

### Batch Flow

1. Read uploaded file — CSV via `pd.read_csv(encoding="utf-8", encoding_errors="replace")`, TXT as one line per entry (UTF-8 with replacement for invalid bytes)
2. Maximum 100 rows per batch. If file exceeds this, show `st.warning()` and truncate
3. Skip empty rows
4. For each row, run the same translation function
5. Show `st.progress()` bar during batch processing
6. Collect results into a DataFrame with columns: `original`, `translated`
7. Display with `st.dataframe()`, offer `st.download_button()` for CSV export

### No Streaming

`model.generate()` returns the full output at once. Streaming with local Transformers adds complexity (TextIteratorStreamer + threading) for minimal benefit on a translation tool where outputs are short.

## Supported Languages

The Water variant is optimized for European and Asia-Pacific languages, but the underlying model supports all 70+ Tiny Aya languages. The app exposes the Water-optimized subset since those will produce the best results with this variant. Users needing other languages should use the Global, Fire, or Earth variants.

**European (31):** English, Dutch, French, Italian, Portuguese, Romanian, Spanish, Czech, Polish, Ukrainian, Russian, Greek, German, Danish, Swedish, Norwegian, Catalan, Galician, Welsh, Irish, Basque, Croatian, Latvian, Lithuanian, Slovak, Slovenian, Estonian, Finnish, Hungarian, Serbian, Bulgarian

**Asia-Pacific (12):** Chinese, Japanese, Korean, Tagalog, Malay, Indonesian, Javanese, Khmer, Thai, Lao, Vietnamese, Burmese

## File Structure

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Single-file app with all logic |
| `test_streamlit_app.py` | Pytest tests for translation and file processing functions |
| `pyproject.toml` | uv-managed project with all dependencies |
| `.env.example` | Documents configurable environment variables |

## Dependencies

- `streamlit` — UI framework
- `transformers` — model loading and inference
- `torch` — PyTorch backend
- `accelerate` — required for `device_map` and efficient model loading
- `pandas` — CSV handling for batch mode
- `python-dotenv` — env config loading
- Dev: `ruff`, `ty`, `pytest`

## `.env.example`

```
MODEL_ID=CohereLabs/tiny-aya-water
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=700
DEVICE=cpu
TOP_P=0.95
MAX_BATCH_ROWS=100
```

## Testable Functions

Functions extracted from the Streamlit UI for independent testing:

- `build_translation_prompt(text, source_lang, target_lang)` — returns the messages list
- `extract_translation(decoded_text)` — trims whitespace from decoded output (decoding uses `skip_special_tokens=True`, so this is mainly cleanup)
- `parse_uploaded_file(file, column)` — returns a list of strings from CSV or TXT
- `translate_text(text, source_lang, target_lang, model, tokenizer, temperature, max_tokens)` — end-to-end translation
