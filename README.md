# Tiny Aya Global Translate

Translate text and voice across 67 languages — all running privately on your Mac. Powered by [mlx-community/tiny-aya-global-8bit-mlx](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx) and [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

## Features

- Side-by-side text translation
- Swap and download controls
- Voice translation: record from mic or upload an audio file
- Auto-transcribed in 14 source languages via [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and auto-translated
- 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa
- 8-bit quantized MLX inference on Apple Silicon
- Local inference — no API key required

## Prerequisites

- Apple Silicon Mac
- 16 GB+ RAM recommended (two MLX models load to ~6 GB)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

First run downloads two models: tiny-aya-global (~1.7 GB) and Cohere Transcribe (~4.1 GB). Voice recording uses your browser's microphone — you'll be asked to grant permission on first use. To tune the models or sampling parameters, edit the constants at the top of `streamlit_app.py`.

## Development

```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py                             # type check
```

## License

This app bundles two models:

- tiny-aya-global — [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial only)
- Cohere Transcribe — [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The combined product is non-commercial because of the tiny-aya-global license.
