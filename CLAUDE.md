## Project

Streamlit app for translating text across 67 languages using the `mlx-community/tiny-aya-global-8bit-mlx` model with local MLX inference on Apple Silicon.

## Stack

- Python 3.13+ with uv for project management
- Streamlit for UI
- mlx-lm for translation inference on Apple Silicon
- mlx-speech for ASR (Cohere Transcribe) inference on Apple Silicon

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
- ASR (Cohere Transcribe) loads via `@st.cache_resource def load_asr_model()`, mirroring `load_model()`. Two cached MLX models live in memory for the session lifetime.
- `load_asr_model` calls `huggingface_hub.snapshot_download(repo_id=ASR_MODEL_ID)` first, then passes the local snapshot path joined with `ASR_MODEL_SUBDIR` (`"mlx-int8"`) to `CohereAsrModel.from_path`. `from_path` is just an alias for `from_dir` and expects a literal directory containing `config.json`, not an HF repo id.
- `ASR_LANGUAGE_CODES` is the single source of truth for which `LANGUAGES` entries the audio uploader supports — 14 of the 67. Other source languages disable the uploader and surface an `st.info` banner explaining why.
- The audio uploader (`st.file_uploader`) sits between the language bar and the warning slot. Placement is constrained: the transcription block writes to `st.session_state.translate_input` before the input `text_area` renders, because Streamlit raises `StreamlitAPIException` if you mutate a keyed widget's session state after it's been rendered.
- `transcribe_audio` accepts raw bytes (not Streamlit's `UploadedFile`) so it's trivially mockable in tests; call sites pass `uploaded.getvalue()`.
- UI tests for the file uploader drive transcription by setting `st.session_state.audio_file` and `st.session_state._do_transcribe` directly, then calling `at.run(...)` — AppTest has no documented public API for `st.file_uploader.set_value(...)`.
- AppTest fixtures and helpers patch `huggingface_hub.snapshot_download`, `mlx_speech.generation.CohereAsrModel.from_path`, `mlx_lm.load`, and (for transcription tests) `soundfile.read` — all at the upstream library level, not at `streamlit_app`, because AppTest runs the script via `exec()`.
- The app bundles two models: `mlx-community/tiny-aya-global-8bit-mlx` (translation, CC-BY-NC, non-commercial only) and `mlx-community/cohere-transcribe-03-2026-mlx-8bit` (ASR, Apache 2.0). The combined product is non-commercial because of the translation model's license.
