## Project

Streamlit app for translating text across 67 languages using `mlx-community/tiny-aya-global-8bit-mlx` with local MLX inference on Apple Silicon. Voice and audio-file input (mic recording or upload) is transcribed via Cohere Transcribe (14 source languages) and auto-translated.

## Stack

- Python 3.13+ with uv for project management
- Streamlit for UI
- mlx-lm for translation inference on Apple Silicon
- mlx-speech for ASR (Cohere Transcribe) inference on Apple Silicon
- mlx for VAD inference (Silero VAD v6) on Apple Silicon

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `vad.py` — vendored MLX forward pass for Silero VAD v6 (silence detection)
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface

## Commands

```bash
uv run streamlit run streamlit_app.py                        # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py vad.py                      # type check
```

## Conventions

- Pure functions are defined above `import streamlit` with deferred imports for heavy deps (`mlx_lm`, `mlx_speech`) inside their bodies, so tests can patch upstream libraries without loading the model stack
- Config is hardcoded as module-level constants (`MODEL_ID`, `ASR_MODEL_ID`, `ASR_MODEL_SUBDIR`, `ASR_LANGUAGE_CODES`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`, `VAD_MODEL_ID`, `VAD_THRESHOLD`, `VAD_PAD_SECONDS`) at the top of `streamlit_app.py`
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- Swap button (`:material/swap_horiz:`, `type="tertiary"`, `help=` tooltip) flips languages via `st.session_state` and moves output into input
- Audio inputs are `st.columns(2)`: `st.audio_input` (mic, left) and `st.file_uploader` (right), between the language bar and the warning slot; both widgets share `uploader_disabled` / `uploader_help`
- The transcription block must run before the input `text_area` renders since Streamlit raises `StreamlitAPIException` if you mutate a keyed widget's session state after it's been rendered
- Source-tag pattern: `request_mic_transcribe` and `request_upload_transcribe` set `_do_transcribe = True` and `_transcribe_source = "mic" | "upload"`; the transcription block reads the tag to choose between `st.session_state.mic_input` and `st.session_state.audio_file`
- Auto-translate chain: on a successful transcription, the success branch sets `_do_translate = True` (gated on `model_loaded`); the existing translation processing block consumes it later in the same script pass
- Don't pre-initialize a session-state key that a widget claims via `key=` — Streamlit raises `StreamlitValueAssignmentNotAllowedError`. `mic_input` and `audio_file` are never set in the session-state defaults; non-widget flags like `_transcribe_source` are initialized normally
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled)
- Translate button (`type="primary"`, `use_container_width=True`) uses a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Download button (`type="secondary"`, `use_container_width=True`) uses `st.download_button` to save translation as `translation.txt`
- Controls row is `st.columns(2)`, mirroring the side-by-side input/output panels
- `ASR_LANGUAGE_CODES` is the single source of truth for which 14 of the 67 `LANGUAGES` support audio input; unsupported source languages disable both audio widgets and show an `st.info` banner explaining why
- Translation model loads via `@st.cache_resource def load_model()` using `mlx_lm.load`
- ASR model loads via `@st.cache_resource def load_asr_model()`, which calls `huggingface_hub.snapshot_download(repo_id=ASR_MODEL_ID)` then passes `local_dir / ASR_MODEL_SUBDIR` (`"mlx-int8"`) to `CohereAsrModel.from_path` — `from_path` expects a literal directory containing `config.json`, not an HF repo id
- VAD model loads via `@st.cache_resource def load_vad_model()`, which calls `huggingface_hub.snapshot_download(repo_id=VAD_MODEL_ID)` then passes the resulting directory to `vad.load_vad`. Returns a flat `dict[str, mx.array]` weight dict — there is no model object/class
- Audio pipeline is three pure functions: `decode_audio(bytes) -> ndarray`, `detect_speech(ndarray, vad_model) -> (start_sec, end_sec) | None`, `transcribe_audio(ndarray, lang, asr_model) -> str`. Decoding is shared upstream of both VAD and ASR
- VAD load failure is graceful degradation, not a gate — audio inputs stay enabled and transcription proceeds without trimming or empty-recording rejection. The `vad_loaded` flag controls whether the VAD branch runs in the transcription block
- `translate_text` builds a chat prompt, applies the tokenizer chat template, samples via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `decode_audio` accepts raw bytes (not Streamlit's `UploadedFile`) so it's trivially mockable; decodes via `soundfile`, downmixes stereo with `mean(axis=1)`, and resamples to 16 kHz with `numpy.interp`. Returns a mono float32 ndarray that both `detect_speech` and `transcribe_audio` consume
- `transcribe_audio` takes the decoded ndarray directly (decoupled from upload handling) and calls the ASR model's `transcribe(audio=, sample_rate=16000, language=)`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- UI tests use `streamlit.testing.v1.AppTest`; mocks target upstream libraries (`mlx_lm`, `mlx_speech`, `huggingface_hub`, `soundfile`) because AppTest runs scripts via `exec()`; widgets without named accessors (audio input, file uploader, download button) are accessed via `at.get(...)[0]`
- UI tests drive transcription via `_drive_transcription` (sets `audio_file` + `_transcribe_source = "upload"`) or `_drive_mic_transcription` (sets `mic_input` + `_transcribe_source = "mic"`), then call `at.run(...)` — AppTest has no public API for either widget's value setter
- The app bundles three models: `mlx-community/tiny-aya-global-8bit-mlx` (translation, CC-BY-NC, non-commercial only), `mlx-community/cohere-transcribe-03-2026-mlx-8bit` (ASR, Apache 2.0), and `mlx-community/silero-vad-v6` (VAD, MIT); the combined product is non-commercial because of the translation model
