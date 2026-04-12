## Project

Streamlit app for translating text across 67 languages using the `mlx-community/tiny-aya-global-8bit-mlx` model with local MLX inference on Apple Silicon.

## Stack

- Python 3.12+ with uv for project management
- Streamlit for UI
- mlx-lm for inference on Apple Silicon
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `.env.example` — runtime configuration defaults
- `docs/` — design specs and implementation plans

## Commands

```bash
uv run streamlit run streamlit_app.py                        # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py                             # type check
```

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- All utility buttons use Material Icons via the `icon` parameter and `help=` for hover tooltips
- Swap button (`:material/swap_horiz:`, tertiary) flips languages via `st.session_state` and moves output into input
- Translate button uses `type="primary"` with a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled) with a controls row (Translate, clear, copy, download)
- Copy button uses `subprocess` + `/usr/bin/pbcopy` for plain-text clipboard with `st.toast("Translation copied")` on success
- Download button uses `st.download_button` to save translation as `translation.txt`
- `translate_text` builds a chat prompt, formats it with `tokenizer.apply_chat_template`, creates a sampler via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- Model loaded once via `@st.cache_resource` using `mlx_lm.load`; runs on Apple Silicon only
- Config loaded from `.env` via python-dotenv with sensible defaults
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- The `mlx-community/tiny-aya-global-8bit-mlx` model is licensed CC-BY-NC (non-commercial use only)
