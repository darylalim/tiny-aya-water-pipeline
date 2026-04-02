# Migration to tiny-aya-global-8bit-mlx

## Summary

Replace the current `transformers` + PyTorch inference backend with Apple's `mlx-lm` framework, and switch from `CohereLabs/tiny-aya-water` (43 languages) to `mlx-community/tiny-aya-global-8bit-mlx` (67 languages). The app becomes Mac-only (Apple Silicon).

## Motivation

There is no MLX version of tiny-aya-water available on HuggingFace. The global variant provides broader language coverage (67 vs 43) and 8-bit quantization for efficient Apple Silicon inference via MLX.

## Dependencies

**Remove:** `torch`, `transformers`, `accelerate`
**Add:** `mlx-lm`
**Keep:** `streamlit`, `python-dotenv`, `python-docx`, `python-pptx`, `openpyxl`, `pymupdf`

## Model loading & inference

- `load_model()` calls `mlx_lm.load("mlx-community/tiny-aya-global-8bit-mlx")`, returns `(model, tokenizer)`, cached with `@st.cache_resource`
- `translate_text()` builds the chat prompt via `tokenizer.apply_chat_template()` (returns a string), then calls `mlx_lm.generate(model, tokenizer, prompt=..., max_tokens=..., temp=...)` which returns a string directly
- `clean_model_output()` stays as-is (strip whitespace from generated text)

**Remove entirely:**
- `detect_device()` and `select_dtype()` — no device/dtype logic needed
- `DEVICE` config and `.env` `DEVICE=auto` setting
- All `import torch` statements
- The `hasattr(inputs, "keys")` / `BatchEncoding` branching in `_generate()`
- `_generate()` helper — logic folds into `translate_text()` or a simplified helper

## Config & environment

- `MODEL_ID` default: `"mlx-community/tiny-aya-global-8bit-mlx"`
- `DEVICE` removed from config and `.env.example`
- `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P` stay
- Loading spinner: `"Loading model..."` (no device name)

## Languages (67 total)

Region names and groupings per Table 1 of the Tiny Aya research paper (arXiv:2603.11510, section 2.3.3).

### Europe (31)
English, Dutch, French, Italian, Portuguese, Romanian, Spanish, Czech, Polish, Ukrainian, Russian, Greek, German, Danish, Swedish, Bokmål, Catalan, Galician, Welsh, Irish, Basque, Croatian, Latvian, Lithuanian, Slovak, Slovenian, Estonian, Finnish, Hungarian, Serbian, Bulgarian

### West Asia (5)
Arabic, Persian, Turkish, Maltese, Hebrew

### South Asia (9)
Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu, Nepali, Urdu

### Asia Pacific (12)
Tagalog, Malay, Indonesian, Vietnamese, Javanese, Khmer, Thai, Lao, Chinese, Burmese, Japanese, Korean

### African (10)
Amharic, Hausa, Igbo, Malagasy, Shona, Swahili, Wolof, Xhosa, Yoruba, Zulu

**Note:** "Norwegian" in the current list becomes "Bokmål" per the paper.

## Project metadata updates

- `pyproject.toml`: update description to reference tiny-aya-global, replace deps
- `CLAUDE.md`: update model name, stack (mlx-lm instead of Transformers + PyTorch), Mac-only, 67 languages
- `.env.example`: remove `DEVICE`, update `MODEL_ID`

## Tests

- `test_streamlit_app.py`: Remove `detect_device`/`select_dtype` tests. Update `translate_text` tests to mock `mlx_lm.generate` (returns string directly) instead of PyTorch tensors. Update language count to 67. Remove `import torch`.
- `test_streamlit_ui.py`: Mock `mlx_lm.load` and `mlx_lm.generate` instead of `transformers.*`. Inference mock helpers simplify (no tensors). Remove `import torch`.
- `test_document.py`: No changes — document processing is independent of the inference backend.

## Unchanged

- `document.py` — entirely untouched
- UI layout, tabs, session state, swap/clear/copy/download logic
- The `translate_fn` callback pattern for document translation
