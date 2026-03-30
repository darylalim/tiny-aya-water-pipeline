## Project

Streamlit app for translating text across 43 European and Asia-Pacific languages using CohereLabs/tiny-aya-water (3.35B parameter multilingual model) with local HuggingFace Transformers inference.

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
- Side-by-side layout with language bar (`[From] [⇄] [To]`) above input text area and disabled `st.text_area()` output with "Translation" placeholder
- Swap button (`⇄`, tertiary type) flips languages and moves output into input via `st.session_state`
- Language selectboxes use the flat `LANGUAGES` list (43 items) with collapsed labels and Streamlit's built-in type-to-search
- Translate button uses `type="primary"` for visual prominence; renders before text panels (Streamlit session state constraint)
- Clear button (`✕`, tertiary) and character count caption below input; copy button (`⧉`, tertiary) below output uses `st.html()` JS clipboard injection
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `transformers` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- `translate_text` handles both plain tensor and `BatchEncoding` returns from `apply_chat_template`
- `clean_model_output` cleans decoded model output
- Device auto-detected (CUDA > MPS > CPU) with optimal dtype (BF16, FP16, FP32); override via `DEVICE` in `.env`
- Model loaded once via `@st.cache_resource` with `dtype` (not deprecated `torch_dtype`); inference runs under `torch.inference_mode()`
- Config loaded from `.env` via python-dotenv with sensible defaults
- License: CC-BY-NC (non-commercial use only)
