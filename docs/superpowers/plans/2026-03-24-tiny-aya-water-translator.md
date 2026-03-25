# Tiny Aya Water Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit translation app using CohereLabs/tiny-aya-water with local HuggingFace Transformers inference, supporting single text and batch file translation across 43 European and Asia-Pacific languages.

**Architecture:** Single-file Streamlit app (`streamlit_app.py`) with testable functions extracted for unit testing. Model loaded once via `@st.cache_resource`. Config from `.env` via `python-dotenv`. TDD approach — tests first for all pure functions.

**Tech Stack:** Streamlit, Transformers, PyTorch, Accelerate, Pandas, python-dotenv, uv, ruff, ty, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-tiny-aya-water-translator-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `streamlit_app.py` | All app logic: config loading, language list, prompt building, translation, file parsing, Streamlit UI |
| `test_streamlit_app.py` | Pytest unit tests for `build_translation_prompt`, `extract_translation`, `parse_uploaded_file`, and `translate_text` (mocked model) |
| `pyproject.toml` | uv-managed project metadata and dependencies |
| `.env.example` | Documents all configurable environment variables with defaults |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore` (update existing)

- [ ] **Step 1: Initialize uv project and add dependencies**

```bash
cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/tiny-aya-water-pipeline"
uv init --name tiny-aya-water-translator
uv add streamlit transformers torch accelerate pandas python-dotenv
uv add --dev ruff ty pytest
```

- [ ] **Step 2: Create `.env.example`**

```
# .env.example — Tiny Aya Water Translator configuration
MODEL_ID=CohereLabs/tiny-aya-water
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=700
DEVICE=cpu
TOP_P=0.95
MAX_BATCH_ROWS=100
```

- [ ] **Step 3: Update `.gitignore`**

Add these entries if not already present:

```
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

- [ ] **Step 4: Verify project setup**

Run: `uv run python -c "import streamlit; import transformers; import torch; import pandas; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore
git commit -m "chore: scaffold project with uv, add dependencies and .env.example"
```

---

### Task 2: Language list and `build_translation_prompt`

**Files:**
- Create: `streamlit_app.py` (initial version with constants and prompt builder)
- Create: `test_streamlit_app.py` (initial version with prompt builder tests)

- [ ] **Step 1: Write failing tests for `build_translation_prompt`**

Create `test_streamlit_app.py`:

```python
from streamlit_app import LANGUAGES, build_translation_prompt


def test_languages_list_has_43_entries() -> None:
    assert len(LANGUAGES) == 43


def test_languages_list_contains_english() -> None:
    assert "English" in LANGUAGES


def test_languages_list_contains_japanese() -> None:
    assert "Japanese" in LANGUAGES


def test_build_translation_prompt_returns_single_message() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_build_translation_prompt_contains_languages() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "English" in content
    assert "French" in content


def test_build_translation_prompt_contains_text() -> None:
    result = build_translation_prompt("Good morning", "English", "Spanish")
    content = result[0]["content"]
    assert "Good morning" in content


def test_build_translation_prompt_instruction() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "Translate" in content
    assert "Output only the translation" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write minimal implementation**

Create `streamlit_app.py` with the language list and prompt builder:

```python
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# -- Config ------------------------------------------------------------------

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

MODEL_ID: str = os.getenv("MODEL_ID", "CohereLabs/tiny-aya-water")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
DEVICE: str = os.getenv("DEVICE", "cpu")
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
MAX_BATCH_ROWS: int = int(os.getenv("MAX_BATCH_ROWS", "100"))

# -- Languages ---------------------------------------------------------------
# Water variant: optimized for European + Asia-Pacific languages.

LANGUAGES: list[str] = [
    # European (31)
    "English",
    "Dutch",
    "French",
    "Italian",
    "Portuguese",
    "Romanian",
    "Spanish",
    "Czech",
    "Polish",
    "Ukrainian",
    "Russian",
    "Greek",
    "German",
    "Danish",
    "Swedish",
    "Norwegian",
    "Catalan",
    "Galician",
    "Welsh",
    "Irish",
    "Basque",
    "Croatian",
    "Latvian",
    "Lithuanian",
    "Slovak",
    "Slovenian",
    "Estonian",
    "Finnish",
    "Hungarian",
    "Serbian",
    "Bulgarian",
    # Asia-Pacific (12)
    "Chinese",
    "Japanese",
    "Korean",
    "Tagalog",
    "Malay",
    "Indonesian",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Vietnamese",
    "Burmese",
]


# -- Pure functions -----------------------------------------------------------


def build_translation_prompt(
    text: str, source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    """Build the chat messages list for a translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Output only the translation, nothing else.\n\n{text}"
            ),
        }
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add language list and build_translation_prompt with tests"
```

---

### Task 3: `extract_translation` and `parse_uploaded_file`

**Files:**
- Modify: `streamlit_app.py` (add two functions)
- Modify: `test_streamlit_app.py` (add tests)

- [ ] **Step 1: Write failing tests for `extract_translation`**

Append to `test_streamlit_app.py`:

```python
from streamlit_app import extract_translation


def test_extract_translation_strips_whitespace() -> None:
    assert extract_translation("  Hello world  ") == "Hello world"


def test_extract_translation_empty_string() -> None:
    assert extract_translation("") == ""


def test_extract_translation_newlines() -> None:
    assert extract_translation("\n\nBonjour\n\n") == "Bonjour"


def test_extract_translation_preserves_inner_whitespace() -> None:
    assert extract_translation("  Hello world  ") == "Hello world"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest test_streamlit_app.py -v -k "extract"`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `extract_translation`**

Add to `streamlit_app.py` after `build_translation_prompt`:

```python
def extract_translation(decoded_text: str) -> str:
    """Clean up decoded model output (skip_special_tokens=True already applied)."""
    return decoded_text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v -k "extract"`
Expected: All 4 tests PASS

- [ ] **Step 5: Write failing tests for `parse_uploaded_file`**

Append to `test_streamlit_app.py`:

```python
from streamlit_app import parse_uploaded_file


def test_parse_uploaded_file_csv_default_column() -> None:
    csv_content = b"text,other\nhello,1\nworld,2\n"
    file = BytesIO(csv_content)
    file.name = "test.csv"
    result = parse_uploaded_file(file, column="text")
    assert result == ["hello", "world"]


def test_parse_uploaded_file_txt() -> None:
    txt_content = b"hello\nworld\n"
    file = BytesIO(txt_content)
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None)
    assert result == ["hello", "world"]


def test_parse_uploaded_file_skips_empty_rows() -> None:
    txt_content = b"hello\n\nworld\n\n"
    file = BytesIO(txt_content)
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None)
    assert result == ["hello", "world"]


def test_parse_uploaded_file_truncates_at_max_rows() -> None:
    lines = "\n".join(f"line{i}" for i in range(200)) + "\n"
    file = BytesIO(lines.encode("utf-8"))
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None, max_rows=100)
    assert len(result) == 100
    assert result[0] == "line0"
    assert result[99] == "line99"


def test_parse_uploaded_file_csv_missing_column() -> None:
    csv_content = b"text,other\nhello,1\n"
    file = BytesIO(csv_content)
    file.name = "test.csv"
    result = parse_uploaded_file(file, column="nonexistent")
    assert result == []
```

Add `from io import BytesIO` to the top of `test_streamlit_app.py` if not already there.

- [ ] **Step 6: Run tests to verify new tests fail**

Run: `uv run pytest test_streamlit_app.py -v -k "parse"`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement `parse_uploaded_file`**

Add to `streamlit_app.py` after `extract_translation`:

```python
def parse_uploaded_file(
    file: BytesIO, column: str | None, max_rows: int = MAX_BATCH_ROWS
) -> list[str]:
    """Extract a list of text strings from an uploaded CSV or TXT file."""
    name: str = getattr(file, "name", "")
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
        except Exception:
            return []
        if column and column in df.columns:
            texts = df[column].astype(str).tolist()
        elif column:
            return []
        else:
            texts = df.iloc[:, 0].astype(str).tolist()
    else:
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        texts = raw.splitlines()

    # Filter empty rows and truncate
    texts = [t for t in texts if t.strip()]
    return texts[:max_rows]
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add extract_translation and parse_uploaded_file with tests"
```

---

### Task 4: `translate_text` with mocked model

**Files:**
- Modify: `streamlit_app.py` (add `translate_text`)
- Modify: `test_streamlit_app.py` (add tests with mock)

- [ ] **Step 1: Write failing tests for `translate_text`**

Append to `test_streamlit_app.py`:

```python
from unittest.mock import MagicMock

import torch

from streamlit_app import translate_text


def test_translate_text_returns_string() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # apply_chat_template returns token IDs
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]

    # model.generate returns a tensor
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    # decode returns the "translated" text
    mock_tokenizer.decode.return_value = "  Bonjour  "

    result = translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.1,
        max_tokens=700,
    )
    assert result == "Bonjour"


def test_translate_text_calls_generate_with_correct_params() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Bonjour"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.3,
        max_tokens=500,
    )

    # Verify generate was called
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 500
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["do_sample"] is True


def test_translate_text_passes_prompt_to_tokenizer() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = "Hola"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="Spanish",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.1,
        max_tokens=700,
    )

    # Verify the prompt was built and passed to apply_chat_template
    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert "English" in messages[0]["content"]
    assert "Spanish" in messages[0]["content"]
    assert "Hello" in messages[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_app.py -v -k "translate_text"`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `translate_text`**

Add to `streamlit_app.py` after `parse_uploaded_file`:

```python
def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model: object,
    tokenizer: object,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Translate text using the model and return the cleaned result."""
    messages = build_translation_prompt(text, source_lang, target_lang)
    input_ids = tokenizer.apply_chat_template(  # type: ignore[union-attr]
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    gen_tokens = model.generate(  # type: ignore[union-attr]
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
    )
    # Decode only the newly generated tokens (skip the input prompt)
    output_tokens = gen_tokens[0][input_ids.shape[-1] :]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)  # type: ignore[union-attr]
    return extract_translation(decoded)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add translate_text function with mocked model tests"
```

---

### Task 5: Streamlit UI — model loading and sidebar

**Files:**
- Modify: `streamlit_app.py` (add Streamlit UI code at bottom)

- [ ] **Step 1: Add model loading function and sidebar**

Append to `streamlit_app.py` after all pure functions, guarded by a Streamlit import:

```python
import streamlit as st


@st.cache_resource
def load_model() -> tuple:
    """Load tokenizer and model once, cached for the session lifetime."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=DEVICE
    )
    return tokenizer, model


# -- Sidebar ------------------------------------------------------------------

st.sidebar.title("Settings")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05
)
max_tokens = st.sidebar.slider(
    "Max New Tokens", min_value=100, max_value=2000, value=DEFAULT_MAX_TOKENS, step=10
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)  \n"
    "License: CC-BY-NC (non-commercial)"
)

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False
```

- [ ] **Step 2: Verify the app starts without error (model download not required for syntax check)**

Run: `uv run python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add model loading with cache and sidebar settings"
```

---

### Task 6: Streamlit UI — single text translation

**Files:**
- Modify: `streamlit_app.py` (add main page UI)

- [ ] **Step 1: Add title, language selectors, and single translation UI**

Append to `streamlit_app.py` after model loading:

```python
# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water Translator")
st.markdown(
    "Translate between 43 European and Asia-Pacific languages using "
    "[CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) "
    "running locally."
)

# Language selectors
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Source Language", LANGUAGES, index=LANGUAGES.index("English"))
with col2:
    target_lang = st.selectbox("Target Language", LANGUAGES, index=LANGUAGES.index("French"))

# Single text translation
input_text = st.text_area("Text to translate", height=150)

if st.button("Translate", disabled=not model_loaded):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_lang == target_lang:
        st.warning("Source and target language are the same.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                input_text, source_lang, target_lang, model, tokenizer, temperature, max_tokens
            )
        st.text_area("Translation", value=result, height=150, disabled=True)
```

- [ ] **Step 2: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add single text translation UI with validation"
```

---

### Task 7: Streamlit UI — batch translation

**Files:**
- Modify: `streamlit_app.py` (add batch section)

- [ ] **Step 1: Add batch translation UI**

Append to `streamlit_app.py`:

```python
# -- Batch Translation --------------------------------------------------------

st.markdown("---")
st.subheader("Batch Translation")

uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    # Column selector for CSV
    column: str | None = None
    if uploaded_file.name.endswith(".csv"):
        preview_df = pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="replace")
        uploaded_file.seek(0)  # Reset for re-read
        column = st.selectbox("Column to translate", preview_df.columns.tolist())

    if st.button("Translate File", disabled=not model_loaded):
        if source_lang == target_lang:
            st.warning("Source and target language are the same.")
        else:
            texts = parse_uploaded_file(uploaded_file, column=column)
            if not texts:
                st.warning("No text found in the uploaded file.")
            else:
                if len(texts) >= MAX_BATCH_ROWS:
                    st.warning(
                        f"File exceeds {MAX_BATCH_ROWS} rows. Only the first {MAX_BATCH_ROWS} will be translated."
                    )
                translations: list[str] = []
                progress = st.progress(0)
                for i, text in enumerate(texts):
                    translated = translate_text(
                        text, source_lang, target_lang, model, tokenizer, temperature, max_tokens
                    )
                    translations.append(translated)
                    progress.progress((i + 1) / len(texts))

                result_df = pd.DataFrame({"original": texts, "translated": translations})
                st.dataframe(result_df)

                csv_output = result_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_output,
                    file_name="translations.csv",
                    mime="text/csv",
                )
```

- [ ] **Step 2: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add batch file translation with progress bar and CSV download"
```

---

### Task 8: Lint, format, and type check

**Files:**
- Modify: `streamlit_app.py` (fix any lint/type issues)
- Modify: `test_streamlit_app.py` (fix any lint/type issues)

- [ ] **Step 1: Run ruff check and fix**

Run: `uv run ruff check --fix streamlit_app.py test_streamlit_app.py`
Expected: No remaining errors (or fix any that remain)

- [ ] **Step 2: Run ruff format**

Run: `uv run ruff format streamlit_app.py test_streamlit_app.py`
Expected: Files formatted

- [ ] **Step 3: Run ty type check**

Run: `uv run ty check streamlit_app.py`
Expected: No errors (or fix any that appear)

- [ ] **Step 4: Run all tests**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "chore: lint, format, and type check with ruff and ty"
```

---

### Task 9: Final verification

- [ ] **Step 1: Verify all files exist**

Run: `ls -la streamlit_app.py test_streamlit_app.py pyproject.toml .env.example`
Expected: All 4 files listed

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full lint suite**

Run: `uv run ruff check streamlit_app.py test_streamlit_app.py && uv run ruff format --check streamlit_app.py test_streamlit_app.py`
Expected: No errors

- [ ] **Step 4: Verify app syntax is valid**

Run: `uv run python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Output checklist**

- [ ] `streamlit_app.py` — single-file app with type annotations and inline comments
- [ ] `test_streamlit_app.py` — pytest unit tests for all testable functions
- [ ] `pyproject.toml` — uv-managed project with all dependencies
- [ ] `.env.example` — documents all configurable environment variables with defaults
- [ ] ruff check and ruff format pass clean
- [ ] All pytest tests pass
