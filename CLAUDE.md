## Project

Streamlit app for translating text and documents across 67 languages using mlx-community/tiny-aya-global-8bit-mlx (8-bit quantized multilingual model) with local MLX inference on Apple Silicon. Supports .docx, .pptx, .xlsx, and .pdf files with formatting preserved.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), mlx-lm (inference on Apple Silicon)
- python-docx, python-pptx, openpyxl, pymupdf (document processing)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — main app: config, pure functions, tabbed Streamlit UI (Text + Documents)
- `document.py` — document processing: per-format extract/rebuild + translate_document coordinator
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `test_document.py` — pytest unit tests for document processing
- `.env.example` — configurable environment variables
- `docs/` — design specs and implementation plans

## Commands

```bash
uv run streamlit run streamlit_app.py   # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py test_document.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- UI uses `st.tabs` with "Text" and "Documents" tabs; each tab has its own language bar with independent selection (`source_lang`/`target_lang` vs `doc_source_lang`/`doc_target_lang`)
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- All utility buttons use Material Icons via the `icon` parameter and `help=` for hover tooltips
- Swap button (`:material/swap_horiz:`, tertiary) flips languages via `st.session_state`; Text tab swap also moves output into input
- Translate buttons use `type="primary"` with callback + flag pattern (`_do_translate` / `_do_translate_doc`) and `st.rerun()` to update output
- Text tab: side-by-side input/output `st.text_area()` (output bound via `value=`, disabled) with controls row (Translate, clear, copy, download)
- Copy button uses `subprocess` + `/usr/bin/pbcopy` for plain-text clipboard with `st.toast("Translation copied")` on success
- Download button uses `st.download_button` to save translation as `translation.txt`
- Documents tab: file uploader (10 MB limit, .docx/.pptx/.xlsx/.pdf) → Translate button → download button with output filename `{target_lang}_{original_filename}`
- Document translation uses `document.py` with per-format `extract_segments_*` / `rebuild_document_*` pairs and a `translate_document` coordinator that accepts a `translate_fn` callback (no Streamlit dependency)
- Supported formats: .docx (python-docx), .pptx (python-pptx), .xlsx (openpyxl), .pdf (pymupdf/fitz — best-effort layout preservation)
- `_replace_paragraph_text` helper shared by DOCX and PPTX: replaces text preserving first run's formatting
- `translate_text` builds a chat prompt, formats it with `tokenizer.apply_chat_template`, creates a sampler via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- Model loaded once via `@st.cache_resource` using `mlx_lm.load`; runs on Apple Silicon only
- Config loaded from `.env` via python-dotenv with sensible defaults
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- License: CC-BY-NC (non-commercial use only)
