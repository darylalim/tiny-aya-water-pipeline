# Side-by-Side Translate UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tabbed Translate/Summarize layout with a single-page, side-by-side translation interface with an interactive swap button.

**Architecture:** Remove `st.tabs`, remove all Summarize UI code, restructure the Translate UI into a language bar (selectboxes + swap button) above two side-by-side text areas (input + disabled output). Use `st.session_state` to manage swap behavior. Keep all pure functions intact.

**Tech Stack:** Streamlit (columns, session_state, text_area), pytest, AppTest

---

### Task 1: Add UI tests for the new side-by-side layout

**Files:**
- Modify: `test_streamlit_ui.py:1-273`

Replace the entire UI test file. The new tests cover: language defaults, swap button, side-by-side panels, translate flow, validation, and model load failure. All summarize tests are removed. Tab indexing is removed since there are no tabs.

Note: These tests will fail until Task 2 implements the new UI. That is expected — we write tests first, then make them pass.

- [ ] **Step 1: Write the new UI test file**

Replace the full contents of `test_streamlit_ui.py` with:

```python
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> None:
    """Re-run the app with simple mocks (no generate chain needed)."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        app.run(timeout=60)


def _make_inference_mocks(decode_result: str) -> tuple[MagicMock, MagicMock]:
    """Return mocks configured for a successful _generate call."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    final_model = mock_model.to.return_value.eval.return_value
    final_model.device = torch.device("cpu")
    final_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = decode_result

    return mock_tokenizer, mock_model


def _run_inference_test(input_text: str, decode_result: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    mock_tokenizer, mock_model = _make_inference_mocks(decode_result)
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value(input_text)
        at.button("Translate").click()
        at.run(timeout=60)
    return at


# -- Language defaults ---------------------------------------------------------


def test_source_language_default(app: AppTest) -> None:
    assert app.selectbox[0].value == "English"


def test_target_language_default(app: AppTest) -> None:
    assert app.selectbox[1].value == "French"


# -- Swap button ---------------------------------------------------------------


def test_swap_button_exists(app: AppTest) -> None:
    assert app.button("⇄") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("⇄").click()
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_moves_output_to_input(app: AppTest) -> None:
    """After translating, swap should move the output into the input field."""
    mock_tokenizer, mock_model = _make_inference_mocks("Bonjour")
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("Translate").click()
        at.run(timeout=60)

        # Swap
        at.button("⇄").click()
        at.run(timeout=60)

    # Input should now contain the previous output
    assert at.text_area[0].value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


# -- Text panels ---------------------------------------------------------------


def test_input_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == "Type or paste your text here..."


def test_output_text_area_is_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("Translate") is not None


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    assert at.text_area[1].value == "Bonjour"


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("Translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("Translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)


# -- Language switching --------------------------------------------------------


def test_change_source_language(app: AppTest) -> None:
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "Spanish"


def test_change_target_language(app: AppTest) -> None:
    app.selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[1].value == "Spanish"


# -- Model load failure --------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("download failed"),
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_model_load_failure_disables_translate_button() -> None:
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("download failed"),
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("Translate").disabled
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: Most tests FAIL because the app still uses tabs and the old layout. This confirms the tests are testing the new behavior.

- [ ] **Step 3: Commit the failing tests**

```bash
git add test_streamlit_ui.py
git commit -m "test: rewrite UI tests for side-by-side translate layout"
```

---

### Task 2: Rewrite the Streamlit UI for side-by-side translate

**Files:**
- Modify: `streamlit_app.py:239-314`

Replace everything from the `# -- Main page` comment (line 239) through end of file. Remove the tabs, remove summarize UI, add session state management, swap button, and side-by-side text areas.

- [ ] **Step 1: Replace the UI section of streamlit_app.py**

Replace lines 239-314 (everything from `# -- Main page` to end of file) with:

```python
# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water")
st.markdown("Translate text — running privately on your computer.")

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model, _device, _dtype = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    tokenizer, model = None, None
    model_loaded = False

# -- Session state defaults ---------------------------------------------------

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "French"
if "translate_input" not in st.session_state:
    st.session_state.translate_input = ""
if "translate_output" not in st.session_state:
    st.session_state.translate_output = ""


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


# -- Language bar -------------------------------------------------------------

col_from, col_swap, col_to = st.columns([5, 1, 5])
with col_from:
    source_lang = st.selectbox(
        "From",
        LANGUAGES,
        key="source_lang",
    )
with col_swap:
    st.html(
        "<div style='padding-top:1.8rem'></div>"
    )
    st.button("⇄", on_click=swap_languages)
with col_to:
    target_lang = st.selectbox(
        "To",
        LANGUAGES,
        key="target_lang",
    )

# -- Side-by-side text panels -------------------------------------------------

col_input, col_output = st.columns(2)
with col_input:
    translate_input = st.text_area(
        "Input",
        placeholder="Type or paste your text here...",
        height=200,
        key="translate_input",
        label_visibility="collapsed",
    )
with col_output:
    st.text_area(
        "Output",
        value=st.session_state.translate_output,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

# -- Translate button ---------------------------------------------------------

if st.button("Translate", disabled=not model_loaded):
    if not translate_input.strip():
        st.warning("Please enter some text first.")
    elif source_lang == target_lang:
        st.warning("Please pick two different languages.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                translate_input,
                source_lang,
                target_lang,
                model,
                tokenizer,
            )
        st.session_state.translate_output = result
        st.rerun()
```

- [ ] **Step 2: Run all UI tests to verify they pass**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: All tests PASS.

- [ ] **Step 3: Run unit tests to verify nothing is broken**

Run: `uv run pytest test_streamlit_app.py -v`

Expected: All tests PASS (pure functions are untouched).

- [ ] **Step 4: Run linter and formatter**

Run: `uv run ruff check --fix . && uv run ruff format .`

Expected: Clean or auto-fixed.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: replace tabs with side-by-side translate UI and swap button"
```

---

### Task 3: Update documentation

**Files:**
- Modify: `README.md:1-44`
- Modify: `CLAUDE.md:1-41`

Update both files to reflect the removal of summarization from the UI and the new side-by-side layout.

- [ ] **Step 1: Update README.md**

Replace the full contents of `README.md` with:

```markdown
# Tiny Aya Water

Translate text across 43 languages — all running privately on your computer. Powered by [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water).

## Features

- Side-by-side translation with swap button
- 43 European and Asia-Pacific languages with type-to-search
- Auto-detects GPU (NVIDIA/Apple Silicon) or CPU for best performance
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
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## License

The tiny-aya-water model is licensed [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial).
```

- [ ] **Step 2: Update CLAUDE.md**

Replace the full contents of `CLAUDE.md` with:

```markdown
## Project

Streamlit app for translating text across 43 European and Asia-Pacific languages using CohereLabs/tiny-aya-water (3.35B parameter multilingual model) with local HuggingFace Transformers inference.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), Transformers + PyTorch (inference)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — single-file app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for all pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `.env.example` — configurable environment variables
- `docs/` — design specs and implementation plans

## Commands

```bash
uv run streamlit run streamlit_app.py   # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
```

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- Side-by-side layout with language bar (`[From] [⇄] [To]`) above two equal text area columns (input + disabled output)
- Swap button (`⇄`) flips languages and moves output into input via `st.session_state`
- Language selectboxes use the flat `LANGUAGES` list (43 items) with Streamlit's built-in type-to-search
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `transformers` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- `translate_text` and `summarize_text` handle both plain tensor and `BatchEncoding` returns from `apply_chat_template`
- `clean_model_output` is the shared output cleanup function for both tasks
- Device auto-detected (CUDA > MPS > CPU) with optimal dtype (BF16, FP16, FP32); override via `DEVICE` in `.env`
- Model loaded once via `@st.cache_resource` with `dtype` (not deprecated `torch_dtype`); inference runs under `torch.inference_mode()`
- Config loaded from `.env` via python-dotenv with sensible defaults
- License: CC-BY-NC (non-commercial use only)
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE.md for side-by-side translate UI"
```
