## Project

Streamlit app for translating text across 67 languages using the `mlx-community/tiny-aya-global-8bit-mlx` model with local MLX inference on Apple Silicon.

## Stack

- Python 3.12+ with uv for project management
- Streamlit for UI
- mlx-lm for inference on Apple Silicon

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface

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
- Config is hardcoded as module-level constants (`MODEL_ID`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`) at the top of `streamlit_app.py`
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- Swap button (`:material/swap_horiz:`, `type="tertiary"`, `help=` tooltip) flips languages via `st.session_state` and moves output into input
- Translate button (`type="primary"`, `use_container_width=True`) uses a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Download button (`type="secondary"`, `use_container_width=True`) uses `st.download_button` to save translation as `translation.txt`
- Controls row is `st.columns(2)`, mirroring the side-by-side input/output panels
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled)
- `translate_text` builds a chat prompt, formats it with `tokenizer.apply_chat_template`, creates a sampler via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- Model loaded once via `@st.cache_resource` using `mlx_lm.load`; runs on Apple Silicon only
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` level (not `streamlit_app`) because AppTest runs scripts via `exec()`; the download button is accessed via `at.get("download_button")[0]` since AppTest has no named accessor for it
- The `mlx-community/tiny-aya-global-8bit-mlx` model is licensed CC-BY-NC (non-commercial use only)
