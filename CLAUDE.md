## Project

Streamlit translation app using CohereLabs/tiny-aya-water (3.35B parameter multilingual model) with local HuggingFace Transformers inference. Supports single text and batch file (CSV/TXT) translation across 43 European and Asia-Pacific languages.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), Transformers + PyTorch (inference), Pandas (batch CSV)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — single-file app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for all pure functions
- `.env.example` — configurable environment variables

## Commands

```bash
uv run streamlit run streamlit_app.py   # run the app
uv run pytest test_streamlit_app.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- Model loaded once via `@st.cache_resource`; BF16 on GPU, FP32 on CPU
- Config loaded from `.env` via python-dotenv with sensible defaults
- License: CC-BY-NC (non-commercial use only)
