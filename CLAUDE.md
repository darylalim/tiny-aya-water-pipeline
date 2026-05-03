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

- Pure functions are defined above `import streamlit` with deferred imports for heavy deps (`mlx_lm`, `mlx_speech`) inside their bodies, so tests can patch upstream libraries without loading the model stack
- Config is hardcoded as module-level constants (`MODEL_ID`, `ASR_MODEL_ID`, `ASR_MODEL_SUBDIR`, `ASR_LANGUAGE_CODES`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`) at the top of `streamlit_app.py`
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- Swap button (`:material/swap_horiz:`, `type="tertiary"`, `help=` tooltip) flips languages via `st.session_state` and moves output into input
- Audio uploader (`st.file_uploader`) sits between the language bar and the warning slot; the transcription block must run before the input `text_area` renders since Streamlit raises `StreamlitAPIException` if you mutate a keyed widget's session state after it's been rendered
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled)
- Translate button (`type="primary"`, `use_container_width=True`) uses a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Download button (`type="secondary"`, `use_container_width=True`) uses `st.download_button` to save translation as `translation.txt`
- Controls row is `st.columns(2)`, mirroring the side-by-side input/output panels
- `ASR_LANGUAGE_CODES` is the single source of truth for which 14 of the 67 `LANGUAGES` the audio uploader supports; unsupported source languages disable the uploader and show an `st.info` banner explaining why
- Translation model loads via `@st.cache_resource def load_model()` using `mlx_lm.load`
- ASR model loads via `@st.cache_resource def load_asr_model()`, which calls `huggingface_hub.snapshot_download(repo_id=ASR_MODEL_ID)` then passes `local_dir / ASR_MODEL_SUBDIR` (`"mlx-int8"`) to `CohereAsrModel.from_path` — `from_path` expects a literal directory containing `config.json`, not an HF repo id
- `translate_text` builds a chat prompt, applies the tokenizer chat template, samples via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `transcribe_audio` accepts raw bytes (not Streamlit's `UploadedFile`) so it's trivially mockable; decodes via `soundfile`, downmixes stereo with `mean(axis=1)`, and resamples to 16 kHz with `numpy.interp`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- UI tests use `streamlit.testing.v1.AppTest`; mocks target upstream libraries (`mlx_lm`, `mlx_speech`, `huggingface_hub`, `soundfile`) because AppTest runs scripts via `exec()`; widgets without named accessors (download button, file uploader) are accessed via `at.get(...)[0]`
- UI tests drive transcription by setting `st.session_state.audio_file` (with a `_FakeUploadedFile`) and `_do_transcribe` directly, then calling `at.run(...)` — AppTest has no public API for `st.file_uploader.set_value(...)`
- The app bundles two models: `mlx-community/tiny-aya-global-8bit-mlx` (translation, CC-BY-NC, non-commercial only) and `mlx-community/cohere-transcribe-03-2026-mlx-8bit` (ASR, Apache 2.0); the combined product is non-commercial
