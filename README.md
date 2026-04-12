# Tiny Aya Global Translate

Translate text across 67 languages — all running privately on your Mac. Powered by [mlx-community/tiny-aya-global-8bit-mlx](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx).

## Features

- Side-by-side text translation
- Swap, clear, copy (plain-text), and download controls
- 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa
- 8-bit quantized MLX inference on Apple Silicon
- Local inference — no API key required

## Prerequisites

- Apple Silicon Mac
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

First run downloads the model (~1.7 GB). To tune the model or sampling parameters, edit the constants at the top of `streamlit_app.py`.

## Development

```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py                             # type check
```

## License

The tiny-aya-global model is licensed [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial).
