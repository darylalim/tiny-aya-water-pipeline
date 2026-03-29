## Project

Streamlit app for translating and summarizing text across 43 European and Asia-Pacific languages using CohereLabs/tiny-aya-water (3.35B parameter multilingual model) with local HuggingFace Transformers inference.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), Transformers + PyTorch (inference)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — single-file app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for all pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `.env.example` — configurable environment variables
- `docs/` — design specs and implementation plans

## Commands

```bash
uv run streamlit run streamlit_app.py   # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- `LANGUAGE_GROUPS` groups the 43 languages by region (European, Asia-Pacific); not used by UI but kept for potential future use
- UI uses `st.tabs` with a compact layout — inline language pickers for Translate, single dropdown for Summarize
- Language selectboxes use the flat `LANGUAGES` list (43 items) with Streamlit's built-in type-to-search
- `select_summary_length` auto-determines summary length from input text size (Short < 500 chars, Medium 500-2000, Long > 2000)
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `transformers` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- `translate_text` and `summarize_text` handle both plain tensor and `BatchEncoding` returns from `apply_chat_template`
- `clean_model_output` is the shared output cleanup function for both tasks
- Device auto-detected (CUDA > MPS > CPU) with optimal dtype (BF16, FP16, FP32); override via `DEVICE` in `.env`
- Model loaded once via `@st.cache_resource` with `dtype` (not deprecated `torch_dtype`); inference runs under `torch.inference_mode()`
- Config loaded from `.env` via python-dotenv with sensible defaults
- License: CC-BY-NC (non-commercial use only)
