# Tiny Aya Water

Translate and summarize across 43 European and Asia-Pacific languages using [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) running locally.

## Features

- Single text translation with language selection
- Cross-lingual summarization with controllable length (short/medium/long)
- Auto-detects CUDA, MPS, and CPU with optimal dtype per device
- Local inference — no API key required

## Prerequisites

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

First run downloads the model (~7 GB). The app auto-detects the best available device (CUDA > MPS > CPU). To override, set `DEVICE=cuda`, `DEVICE=mps`, or `DEVICE=cpu` in `.env`.

## Development

```bash
uv run pytest test_streamlit_app.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## License

The tiny-aya-water model is licensed [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial).
