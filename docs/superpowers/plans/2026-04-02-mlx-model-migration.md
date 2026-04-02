# MLX Model Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace transformers + PyTorch backend with mlx-lm and switch from tiny-aya-water (43 languages) to tiny-aya-global-8bit-mlx (67 languages), making the app Apple Silicon only.

**Architecture:** Single-file app (streamlit_app.py) swaps inference from transformers `AutoModelForCausalLM` to `mlx_lm.load`/`generate`. Model loading stays cached via `@st.cache_resource`. `mlx_lm.generate` returns strings directly — no tensor handling, device placement, or manual decoding. Document processing (document.py) is untouched.

**Tech Stack:** mlx-lm, Streamlit, python-dotenv, python-docx, python-pptx, openpyxl, pymupdf

**Spec:** `docs/superpowers/specs/2026-04-02-mlx-model-migration-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Swap deps: remove torch/transformers/accelerate, add mlx-lm |
| `streamlit_app.py` | Modify | Remove device logic, rewrite inference for mlx-lm, expand languages to 67 |
| `test_streamlit_app.py` | Modify | Remove device/dtype tests, rewrite translate_text tests to mock mlx_lm |
| `test_streamlit_ui.py` | Modify | Replace transformers mocks with mlx_lm mocks |
| `.env.example` | Modify | Remove DEVICE, update MODEL_ID |
| `CLAUDE.md` | Modify | Update model name, stack, language count, conventions |
| `document.py` | No change | |
| `test_document.py` | No change | |

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml:1-17`

- [ ] **Step 1: Update pyproject.toml dependencies and description**

Replace the `[project]` section (lines 1-17):

```toml
[project]
name = "tiny-aya-water-translate"
version = "0.1.0"
description = "Translate text and documents across 67 languages with mlx-community/tiny-aya-global-8bit-mlx on Apple Silicon"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mlx-lm>=0.28.0",
    "openpyxl>=3.1.0",
    "pymupdf>=1.25.0",
    "python-docx>=1.1.0",
    "python-dotenv>=1.2.2",
    "python-pptx>=1.0.0",
    "streamlit>=1.55.0",
]
```

Removes: `torch`, `transformers`, `accelerate`. Adds: `mlx-lm>=0.28.0`.

- [ ] **Step 2: Install new dependencies**

Run: `uv sync`
Expected: Dependencies resolve and install successfully. mlx and mlx-lm are installed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: swap torch/transformers for mlx-lm"
```

---

### Task 2: Update streamlit_app.py for mlx-lm

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Update imports (lines 1-12)**

Replace:

```python
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from dotenv import load_dotenv
```

With:

```python
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
```

Removes `TYPE_CHECKING` and the conditional `import torch`.

- [ ] **Step 2: Update config (lines 20-24)**

Replace:

```python
MODEL_ID: str = os.getenv("MODEL_ID", "CohereLabs/tiny-aya-water")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
DEVICE: str = os.getenv("DEVICE", "auto")
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
```

With:

```python
MODEL_ID: str = os.getenv("MODEL_ID", "mlx-community/tiny-aya-global-8bit-mlx")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
```

Removes `DEVICE`. Updates `MODEL_ID` default.

- [ ] **Step 3: Replace LANGUAGES list (lines 27-75)**

Replace the entire `LANGUAGES` block and its comment with:

```python
# -- Languages ---------------------------------------------------------------
# Global variant: 67 languages across Europe, West Asia, South Asia,
# Asia Pacific, and Africa.

LANGUAGES: list[str] = [
    # Europe (31)
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
    "Bokmål",
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
    # West Asia (5)
    "Arabic",
    "Persian",
    "Turkish",
    "Maltese",
    "Hebrew",
    # South Asia (9)
    "Hindi",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Punjabi",
    "Tamil",
    "Telugu",
    "Nepali",
    "Urdu",
    # Asia Pacific (12)
    "Tagalog",
    "Malay",
    "Indonesian",
    "Vietnamese",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Chinese",
    "Burmese",
    "Japanese",
    "Korean",
    # African (10)
    "Amharic",
    "Hausa",
    "Igbo",
    "Malagasy",
    "Shona",
    "Swahili",
    "Wolof",
    "Xhosa",
    "Yoruba",
    "Zulu",
]
```

67 languages total. "Norwegian" becomes "Bokmål" per the Tiny Aya paper.

- [ ] **Step 4: Replace pure functions section (lines 78-171)**

Remove `detect_device`, `select_dtype`, and `_generate` entirely. Keep `build_translation_prompt` and `clean_model_output` unchanged. Rewrite `translate_text`.

Replace everything from the `# -- Pure functions` comment through the end of `translate_text` with:

```python
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


def clean_model_output(decoded_text: str) -> str:
    """Clean up decoded model output."""
    return decoded_text.strip()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Translate text using the model and return the cleaned result."""
    from mlx_lm import generate

    messages = build_translation_prompt(text, source_lang, target_lang)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=TOP_P,
    )
    return clean_model_output(result)
```

Key changes: `translate_text` now calls `mlx_lm.generate` directly (returns a string). No tensor handling, no device placement, no manual decode.

- [ ] **Step 5: Rewrite load_model (lines 174-186)**

Replace:

```python
import streamlit as st  # noqa: E402


@st.cache_resource
def load_model(device: str) -> tuple:
    """Load tokenizer and model once, cached for the session lifetime."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(device).eval()
    return tokenizer, model, device, dtype
```

With:

```python
import streamlit as st  # noqa: E402


@st.cache_resource
def load_model() -> tuple:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    return model, tokenizer
```

No device parameter, no dtype selection. `mlx_lm.load` handles everything.

- [ ] **Step 6: Update page title and model loading block (lines 191-204)**

Replace:

```python
st.title("Tiny Aya Water Translate")


# -- Model loading ------------------------------------------------------------

try:
    _resolved_device = DEVICE if DEVICE != "auto" else detect_device()
    with st.spinner(f"Loading model on {_resolved_device.upper()}..."):
        tokenizer, model, _device, _dtype = load_model(_resolved_device)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    tokenizer, model = None, None
    model_loaded = False
```

With:

```python
st.title("Tiny Aya Global Translate")


# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model, tokenizer = None, None
    model_loaded = False
```

No device detection. Simplified spinner text. Return order matches `mlx_lm.load`: `(model, tokenizer)`.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: switch to mlx-lm with tiny-aya-global-8bit-mlx (67 languages)"
```

---

### Task 3: Rewrite test_streamlit_app.py

**Files:**
- Modify: `test_streamlit_app.py`

- [ ] **Step 1: Replace test file contents**

Replace the entire file with:

```python
from unittest.mock import MagicMock, patch

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    clean_model_output,
    translate_text,
)

# -- LANGUAGES -----------------------------------------------------------------


def test_languages_list_has_67_entries() -> None:
    assert len(LANGUAGES) == 67


def test_languages_list_contains_english() -> None:
    assert "English" in LANGUAGES


def test_languages_list_contains_japanese() -> None:
    assert "Japanese" in LANGUAGES


# -- build_translation_prompt --------------------------------------------------


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


# -- clean_model_output --------------------------------------------------------


def test_clean_model_output_strips_whitespace() -> None:
    assert clean_model_output("  Hello world  ") == "Hello world"


def test_clean_model_output_empty_string() -> None:
    assert clean_model_output("") == ""


def test_clean_model_output_newlines() -> None:
    assert clean_model_output("\n\nBonjour\n\n") == "Bonjour"


def test_clean_model_output_preserves_inner_whitespace() -> None:
    assert clean_model_output("  Hello   world  ") == "Hello   world"


# -- translate_text ------------------------------------------------------------


@patch("mlx_lm.generate")
def test_translate_text_returns_cleaned_result(mock_generate: MagicMock) -> None:
    mock_generate.return_value = "  Bonjour  "
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    result = translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )
    assert result == "Bonjour"


@patch("mlx_lm.generate")
def test_translate_text_calls_generate_with_correct_params(
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Bonjour"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.3,
        max_tokens=500,
    )

    mock_generate.assert_called_once_with(
        mock_model,
        mock_tokenizer,
        prompt="formatted prompt",
        max_tokens=500,
        temp=0.3,
        top_p=streamlit_app.TOP_P,
    )


@patch("mlx_lm.generate")
def test_translate_text_passes_prompt_to_tokenizer(
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Hola"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="Spanish",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert "English" in messages[0]["content"]
    assert "Spanish" in messages[0]["content"]
    assert "Hello" in messages[0]["content"]


@patch("mlx_lm.generate")
def test_translate_text_uses_default_params(mock_generate: MagicMock) -> None:
    mock_generate.return_value = "Bonjour"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["temp"] == streamlit_app.DEFAULT_TEMPERATURE
    assert call_kwargs["max_tokens"] == streamlit_app.DEFAULT_MAX_TOKENS
```

Removed: all `detect_device`, `select_dtype`, device-override, and tensor-based `translate_text` tests (14 tests removed). Kept: language, prompt, and clean_model_output tests (11 unchanged). Added: 4 mlx-lm-based translate_text tests. Total: 15 tests.

- [ ] **Step 2: Run unit tests**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All 15 tests pass.

- [ ] **Step 3: Commit**

```bash
git add test_streamlit_app.py
git commit -m "test: rewrite unit tests for mlx-lm backend"
```

---

### Task 4: Rewrite test_streamlit_ui.py

**Files:**
- Modify: `test_streamlit_ui.py`

- [ ] **Step 1: Replace test file contents**

Replace the entire file with:

```python
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading."""
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> None:
    """Re-run the app with mocked model loading."""
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        app.run(timeout=60)


def _run_inference_test(input_text: str, generate_result: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.generate", return_value=generate_result),
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
    assert app.button("swap") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_moves_output_to_input() -> None:
    """After translating, swap should move the output into the input field."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.generate", return_value="Bonjour"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("Translate").click()
        at.run(timeout=60)

        # Swap
        at.button("swap").click()
        at.run(timeout=60)

    # Input should now contain the previous output
    assert at.text_area[0].value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


# -- Text panels ---------------------------------------------------------------


def test_input_text_area_has_no_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == ""


def test_output_uses_text_area(app: AppTest) -> None:
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[1].placeholder == "Translation"


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("Translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("Translate").disabled


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
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


# -- Input constraints ---------------------------------------------------------


def test_input_max_chars_enforced(app: AppTest) -> None:
    app.text_area[0].set_value("x" * 5001)
    _rerun_with_mocks(app)

    assert len(app.text_area[0].value) <= 5000


# -- Clear button --------------------------------------------------------------


def test_clear_button_exists(app: AppTest) -> None:
    assert app.button("clear") is not None


def test_clear_button_disabled_when_input_empty(app: AppTest) -> None:
    assert app.button("clear").disabled


def test_clear_button_enabled_when_input_has_text(app: AppTest) -> None:
    app.text_area[0].set_value("Hello")
    _rerun_with_mocks(app)

    assert not app.button("clear").disabled


def test_clear_button_clears_input_and_output() -> None:
    """After translating, clicking clear should clear both panels."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.generate", return_value="Bonjour"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("Translate").click()
        at.run(timeout=60)

        # Click clear
        at.button("clear").click()
        at.run(timeout=60)

    assert at.text_area[0].value == ""
    assert at.text_area[1].value == ""


# -- Copy button ---------------------------------------------------------------


def test_copy_button_exists(app: AppTest) -> None:
    assert app.button("copy") is not None


def test_copy_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.button("copy").disabled


def test_copy_button_enabled_when_output_present() -> None:
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
    assert not at.button("copy").disabled


def test_copy_button_click_no_errors() -> None:
    """Clicking copy with output present should not produce errors."""
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
    at.button("copy").click()
    _rerun_with_mocks(at)

    assert not at.exception


def test_copy_button_shows_toast() -> None:
    """Clicking copy should show a 'Translation copied' toast."""
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
    at.button("copy").click()
    _rerun_with_mocks(at)

    toast_values = [t.value for t in at.toast]
    assert any("Translation copied" in str(v) for v in toast_values)


# -- Output text area ----------------------------------------------------------


def test_output_text_area_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Model load failure --------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_model_load_failure_disables_translate_button() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("Translate").disabled


# -- Documents tab -------------------------------------------------------------


def test_tabs_exist(app: AppTest) -> None:
    assert len(app.tabs) > 0


def test_doc_translate_button_exists(app: AppTest) -> None:
    assert app.button("TranslateDoc") is not None


def test_doc_translate_button_disabled_when_no_file(app: AppTest) -> None:
    assert app.button("TranslateDoc").disabled


def test_doc_translate_button_disabled_when_model_fails() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("TranslateDoc").disabled


def test_doc_language_defaults(app: AppTest) -> None:
    # Doc tab selectboxes are at index 2 and 3 (after text tab's two)
    assert app.selectbox[2].value == "English"
    assert app.selectbox[3].value == "French"


def test_doc_language_independent_from_text_tab(app: AppTest) -> None:
    # Change text tab language
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    # Doc tab languages unchanged
    assert app.selectbox[2].value == "English"
    assert app.selectbox[3].value == "French"


def test_text_language_independent_from_doc_tab(app: AppTest) -> None:
    # Change doc tab language
    app.selectbox[2].set_value("Spanish")
    _rerun_with_mocks(app)

    # Text tab languages unchanged
    assert app.selectbox[0].value == "English"
    assert app.selectbox[1].value == "French"


def test_doc_swap_flips_languages(app: AppTest) -> None:
    app.button("doc_swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[2].value == "French"
    assert app.selectbox[3].value == "English"


def test_doc_file_uploader_exists(app: AppTest) -> None:
    # AppTest in this Streamlit version doesn't expose a typed .file_uploader
    # accessor; fall back to the generic .get() which returns UnknownElement
    # entries for widgets not yet modelled by the testing API.
    assert len(app.get("file_uploader")) >= 1
```

Key changes from the old file:
- Removed `import torch` (no longer a dependency)
- `app` fixture: patches `mlx_lm.load` instead of two `transformers.*` patches
- `_rerun_with_mocks`: patches `mlx_lm.load` only (was two transformers patches)
- `_run_inference_test`: patches `mlx_lm.load` + `mlx_lm.generate` (was transformers patches + tensor mock chain). Parameter renamed `decode_result` -> `generate_result`.
- `_make_inference_mocks` helper: removed entirely (no tensor setup needed)
- All tests using `_make_inference_mocks` inlined with simpler `patch("mlx_lm.generate")` mocks
- Model failure tests: `side_effect` on `mlx_lm.load` instead of `transformers.AutoTokenizer.from_pretrained`
- Total test count unchanged: 32 tests.

- [ ] **Step 2: Run UI tests**

Run: `uv run pytest test_streamlit_ui.py -v`
Expected: All 32 tests pass.

- [ ] **Step 3: Commit**

```bash
git add test_streamlit_ui.py
git commit -m "test: rewrite UI tests for mlx-lm mocking"
```

---

### Task 5: Update project metadata

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update .env.example**

Replace the entire file with:

```
# .env.example — Tiny Aya Global configuration
MODEL_ID=mlx-community/tiny-aya-global-8bit-mlx
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=700
TOP_P=0.95
```

Removes `DEVICE=auto`. Updates `MODEL_ID` and header comment.

- [ ] **Step 2: Update CLAUDE.md**

Replace the entire file with:

```markdown
## Project

Streamlit app for translating text and documents across 67 languages using mlx-community/tiny-aya-global-8bit-mlx (8-bit quantized multilingual model) with local MLX inference on Apple Silicon. Supports .docx, .pptx, .xlsx, and .pdf files with formatting preserved.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), mlx-lm (inference on Apple Silicon)
- python-docx, python-pptx, openpyxl, pymupdf (document processing)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — main app: config, pure functions, tabbed Streamlit UI (Text + Documents)
- `document.py` — document processing: per-format extract/rebuild + translate_document coordinator
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `test_document.py` — pytest unit tests for document processing
- `.env.example` — configurable environment variables
- `docs/` — design specs and implementation plans

## Commands

` ` `bash
uv run streamlit run streamlit_app.py   # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py test_document.py -v  # run tests
uv run ruff check --fix .              # lint
uv run ruff format .                   # format
uv run ty check streamlit_app.py       # type check
` ` `

## Conventions

- Pure functions are defined above `import streamlit` so they can be imported and tested without Streamlit
- UI uses `st.tabs` with "Text" and "Documents" tabs; each tab has its own language bar with independent selection (`source_lang`/`target_lang` vs `doc_source_lang`/`doc_target_lang`)
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- All utility buttons use Material Icons via the `icon` parameter and `help=` for hover tooltips
- Swap button (`:material/swap_horiz:`, tertiary) flips languages via `st.session_state`; Text tab swap also moves output into input
- Translate buttons use `type="primary"` with callback + flag pattern (`_do_translate` / `_do_translate_doc`) and `st.rerun()` to update output
- Text tab: side-by-side input/output `st.text_area()` (output bound via `value=`, disabled) with controls row (Translate, clear, copy, download)
- Copy button uses `subprocess` + `/usr/bin/pbcopy` for plain-text clipboard with `st.toast("Translation copied")` on success
- Download button uses `st.download_button` to save translation as `translation.txt`
- Documents tab: file uploader (10 MB limit, .docx/.pptx/.xlsx/.pdf) → Translate button → download button with output filename `{target_lang}_{original_filename}`
- Document translation uses `document.py` with per-format `extract_segments_*` / `rebuild_document_*` pairs and a `translate_document` coordinator that accepts a `translate_fn` callback (no Streamlit dependency)
- Supported formats: .docx (python-docx), .pptx (python-pptx), .xlsx (openpyxl), .pdf (pymupdf/fitz — best-effort layout preservation)
- `_replace_paragraph_text` helper shared by DOCX and PPTX: replaces text preserving first run's formatting
- `translate_text` builds a chat prompt, formats it with `tokenizer.apply_chat_template`, and generates with `mlx_lm.generate`
- `clean_model_output` cleans decoded model output
- Model loaded once via `@st.cache_resource` using `mlx_lm.load`; runs on Apple Silicon only
- Config loaded from `.env` via python-dotenv with sensible defaults
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- License: CC-BY-NC (non-commercial use only)
```

- [ ] **Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: update project description and conventions for mlx-lm migration"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py test_document.py -v`
Expected: All tests pass (15 unit + 32 UI + ~25 document = ~72 tests).

- [ ] **Step 2: Run linter and formatter**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: No lint errors, all files formatted.

- [ ] **Step 3: Run type checker**

Run: `uv run ty check streamlit_app.py`
Expected: No type errors (or only pre-existing ones unrelated to this change).

- [ ] **Step 4: Verify no torch/transformers references remain**

Run: `grep -rn "import torch\|from torch\|import transformers\|from transformers" streamlit_app.py test_streamlit_app.py test_streamlit_ui.py`
Expected: No matches.
