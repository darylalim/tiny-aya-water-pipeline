# Tiny Aya Global Translate

Translate text across 67 languages — all running privately on your Mac. Powered by [mlx-community/tiny-aya-global-8bit-mlx](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx).

## Features

- Side-by-side text translation with swap, clear, copy (plain-text), and download
- 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa
- 8-bit quantized MLX inference on Apple Silicon
- Local inference — no API key required

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env  # edit as needed
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

First run downloads the model (~1.7 GB). Configuration options are in `.env.example`.

## Development

```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## License

The tiny-aya-global model is licensed [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial).
