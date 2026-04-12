# Remove Documents Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the Documents tab, `document.py`, its tests, and four doc-processing dependencies; flatten the remaining Text UI so it is no longer wrapped in a single-tab `st.tabs` shell.

**Architecture:** Pure deletion + inline unwrap. `streamlit_app.py` currently renders `st.tabs(["Text", "Documents"])` and runs all Text-tab UI inside `with tab_text:`. We remove the `tab_docs` branch entirely, drop `st.tabs`, and unindent the Text-tab body one level. `document.py` and `test_document.py` are deleted outright. `pyproject.toml`, `uv.lock`, `CLAUDE.md`, and `README.md` are updated to match. A single final commit makes the whole change one `git revert` away from recovery.

**Tech Stack:** Python 3.12, Streamlit, mlx-lm, uv, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-04-12-remove-documents-tab-design.md`

**Note on TDD:** This plan is a deletion/refactor, not a feature addition. There are no new behaviors to assert. The existing test suite is the verification — when the doc-specific tests are removed, the remaining `test_streamlit_app.py` + `test_streamlit_ui.py` must still pass unchanged. Each task runs the relevant verification command after the change so regressions are caught immediately.

**Commit strategy:** Per the spec, the entire change lands as a single commit titled `refactor: remove documents tab and doc processing` in Task 8. Earlier tasks do not commit; they leave the working tree dirty for the final commit to collect.

---

## Task 1: Delete `document.py` and `test_document.py`

**Files:**
- Delete: `document.py`
- Delete: `test_document.py`

- [ ] **Step 1: Remove the two files**

Run:
```bash
git rm document.py test_document.py
```

Expected output:
```
rm 'document.py'
rm 'test_document.py'
```

- [ ] **Step 2: Confirm they are gone from the working tree**

Run:
```bash
ls document.py test_document.py 2>&1 || true
```

Expected: both listed as "No such file or directory".

- [ ] **Step 3: Confirm no other source file imports from `document`**

Run:
```bash
grep -rn "from document" --include="*.py" . || echo "no references"
grep -rn "import document" --include="*.py" . || echo "no references"
```

Expected: `no references` for both (the only importer was `streamlit_app.py`'s Documents-tab block, which we remove in Task 3; after Task 3 this check stays clean).

> Do NOT commit yet — changes accumulate for Task 8.

---

## Task 2: Remove doc dependencies from `pyproject.toml` and regenerate lockfile

**Files:**
- Modify: `pyproject.toml` (lines 4 and 7–15)
- Modify: `uv.lock` (regenerated)

- [ ] **Step 1: Update the project description and dependencies**

Replace lines 4 and 7–15 of `pyproject.toml` so the file reads:

```toml
[project]
name = "tiny-aya-global-translate"
version = "0.1.0"
description = "Translate text across 67 languages with mlx-community/tiny-aya-global-8bit-mlx on Apple Silicon"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mlx-lm>=0.28.0",
    "python-dotenv>=1.2.2",
    "streamlit>=1.55.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "ruff>=0.15.7",
    "ty>=0.0.25",
]

[tool.ruff]
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["."]
```

The only changes are: `description` drops "and documents"; `dependencies` drops `openpyxl>=3.1.0`, `pymupdf>=1.25.0`, `python-docx>=1.1.0`, `python-pptx>=1.0.0`. Everything else is identical.

- [ ] **Step 2: Regenerate the lockfile**

Run:
```bash
uv sync
```

Expected: `uv` resolves the reduced dependency set without errors. `uv.lock` is rewritten; the four dropped packages and any transitive-only dependencies (e.g., `lxml` if only used by `python-docx`/`python-pptx`, `XlsxWriter`, `et_xmlfile`, `Pillow` if only via `pymupdf`) are removed from the lockfile. Some transitive deps may stay if they are still required by remaining packages — that is fine.

- [ ] **Step 3: Confirm `pyproject.toml` and `uv.lock` changed**

Run:
```bash
git status pyproject.toml uv.lock
```

Expected: both listed as modified.

> Do NOT commit yet.

---

## Task 3: Remove Documents tab and flatten `streamlit_app.py`

**Files:**
- Modify: `streamlit_app.py` (session state L193–202; callbacks L226–236; body L239–504)

This task is the largest edit. The safest way to do it is a single `Write` of the complete new file so indentation and structure are unambiguous. The reference below is the full expected contents of `streamlit_app.py` after the edit — lines 1–192 (imports, config, languages, pure functions, `st.cache_resource`, title, model-loading block, Text-tab session-state defaults, and callbacks `request_translate` / `swap_languages` / `clear_input`) are unchanged from the current file.

- [ ] **Step 1: Rewrite `streamlit_app.py`**

Overwrite `streamlit_app.py` with the following content:

```python
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# -- Config ------------------------------------------------------------------

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

MODEL_ID: str = os.getenv("MODEL_ID", "mlx-community/tiny-aya-global-8bit-mlx")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
TOP_P: float = float(os.getenv("TOP_P", "0.95"))

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
    return decoded_text.replace("<|END_RESPONSE|>", "").strip()


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
    from mlx_lm.sample_utils import make_sampler

    messages = build_translation_prompt(text, source_lang, target_lang)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temperature, top_p=TOP_P)
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    return clean_model_output(result)


import streamlit as st  # noqa: E402


@st.cache_resource
def load_model() -> tuple:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    return model, tokenizer


# -- Main page ----------------------------------------------------------------

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

# -- Session state defaults ---------------------------------------------------

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "French"
if "translate_input" not in st.session_state:
    st.session_state.translate_input = ""
if "translate_output" not in st.session_state:
    st.session_state.translate_output = ""
if "_do_translate" not in st.session_state:
    st.session_state._do_translate = False


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


def clear_input() -> None:
    """Clear the input and output text."""
    st.session_state.translate_input = ""
    st.session_state.translate_output = ""


# -- Language bar -------------------------------------------------------------

col_from, col_swap, col_to = st.columns([10, 1, 10], vertical_alignment="center")
with col_from:
    source_lang = st.selectbox(
        "From",
        LANGUAGES,
        key="source_lang",
        label_visibility="collapsed",
    )
with col_swap:
    st.button(
        "",
        key="swap",
        icon=":material/swap_horiz:",
        on_click=swap_languages,
        use_container_width=True,
        type="tertiary",
        help="Swap languages",
    )
with col_to:
    target_lang = st.selectbox(
        "To",
        LANGUAGES,
        key="target_lang",
        label_visibility="collapsed",
    )

# -- Warning slot (above panels) ---------------------------------------------

warning_slot = st.container()

# -- Side-by-side text panels ------------------------------------------------

col_input, col_output = st.columns(2)
with col_input:
    translate_input = st.text_area(
        "Input",
        height=300,
        max_chars=5000,
        key="translate_input",
        label_visibility="collapsed",
    )
with col_output:
    st.text_area(
        "Output",
        height=300,
        placeholder="Translation",
        disabled=True,
        value=st.session_state.translate_output,
        label_visibility="collapsed",
    )

# -- Controls row -------------------------------------------------------------

sub_translate, sub_clear, _, sub_copy, sub_download = st.columns(
    [6, 1, 25, 1, 1], vertical_alignment="center", gap="small"
)
with sub_translate:
    st.button(
        "Translate",
        key="Translate",
        on_click=request_translate,
        disabled=not model_loaded,
        type="primary",
    )
with sub_clear:
    st.button(
        "",
        key="clear",
        icon=":material/close:",
        on_click=clear_input,
        disabled=not translate_input.strip(),
        type="tertiary",
        help="Clear source text",
    )
with sub_copy:
    output_has_text = bool(st.session_state.translate_output.strip())
    if st.button(
        "",
        key="copy",
        icon=":material/content_copy:",
        type="tertiary",
        disabled=not output_has_text,
        help="Copy translation",
    ):
        # Use a platform-native CLI to write plain text directly to the
        # system clipboard, bypassing browser iframe clipboard API issues
        # that leak HTML and cause rich-text formatting in rich-text apps.
        _os = platform.system()
        if _os == "Darwin":
            _clip_cmd = ["/usr/bin/pbcopy"]
        elif _os == "Linux":
            _clip_cmd = ["xclip", "-selection", "clipboard"]
        else:
            _clip_cmd = ["clip"]
        try:
            subprocess.run(
                _clip_cmd,
                input=st.session_state.translate_output.encode("utf-8"),
                check=True,
            )
            st.toast("Translation copied")
        except (FileNotFoundError, subprocess.CalledProcessError):
            st.warning("Could not copy to clipboard.")
with sub_download:
    st.download_button(
        "",
        key="download",
        icon=":material/download:",
        data=st.session_state.translate_output,
        file_name="translation.txt",
        mime="text/plain",
        disabled=not output_has_text,
        type="tertiary",
        help="Download translation",
    )

# -- Process translation request (below controls) ---------------------------

if st.session_state._do_translate:
    st.session_state._do_translate = False
    _current_input = st.session_state.translate_input
    if not _current_input.strip():
        warning_slot.warning("Please enter some text first.")
    elif st.session_state.source_lang == st.session_state.target_lang:
        warning_slot.warning("Please pick two different languages.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                _current_input,
                st.session_state.source_lang,
                st.session_state.target_lang,
                model,
                tokenizer,
            )
        st.session_state.translate_output = result
        st.rerun()  # Re-render to update the already-rendered output text_area
```

Changes from the original:
- Removed doc-only session-state defaults `doc_source_lang`, `doc_target_lang`, `doc_translated_bytes`, `doc_translated_filename`, `_do_translate_doc`.
- Removed callbacks `swap_doc_languages` and `request_translate_doc`.
- Removed `tab_text, tab_docs = st.tabs(["Text", "Documents"])` and the entire `with tab_docs:` block.
- Unindented the `with tab_text:` body one level so the language bar, text panels, controls row, and translate-processing block run at module scope.
- No changes to imports, config, `LANGUAGES`, pure functions, `load_model`, `st.title`, model-loading, or Text-tab callbacks.

- [ ] **Step 2: Verify the file parses**

Run:
```bash
uv run python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Verify there are no remaining references to the removed symbols**

Run:
```bash
grep -n "doc_source_lang\|doc_target_lang\|doc_translated_bytes\|doc_translated_filename\|_do_translate_doc\|swap_doc_languages\|request_translate_doc\|tab_docs\|st.tabs\|translate_document" streamlit_app.py || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Run the pure-function unit tests**

Run:
```bash
uv run pytest test_streamlit_app.py -v
```

Expected: all tests pass (they do not depend on the UI changes).

> Do NOT commit yet.

---

## Task 4: Remove Documents-tab tests from `test_streamlit_ui.py`

**Files:**
- Modify: `test_streamlit_ui.py` (remove the "Documents tab" section, current L271–332)

- [ ] **Step 1: Delete the Documents-tab test section**

In `test_streamlit_ui.py`, delete every line from the section marker:

```python
# -- Documents tab -------------------------------------------------------------
```

through the end of the file, including all of these test functions:
- `test_tabs_exist`
- `test_doc_translate_button_exists`
- `test_doc_translate_button_disabled_when_no_file`
- `test_doc_translate_button_disabled_when_model_fails`
- `test_doc_language_defaults`
- `test_doc_language_independent_from_text_tab`
- `test_text_language_independent_from_doc_tab`
- `test_doc_swap_flips_languages`
- `test_doc_file_uploader_exists`

After the edit, the file should end with the "Model load failure" section (the last remaining function is `test_model_load_failure_disables_translate_button`). Leave a single trailing newline at end of file.

- [ ] **Step 2: Verify no doc references remain in the UI test file**

Run:
```bash
grep -n "doc_\|TranslateDoc\|tab_docs\|tabs_exist\|file_uploader" test_streamlit_ui.py || echo "clean"
```

Expected: `clean`.

- [ ] **Step 3: Run the full UI test suite**

Run:
```bash
uv run pytest test_streamlit_ui.py -v
```

Expected: all remaining tests pass. If any Text-tab test fails due to widget-index drift (unlikely, since Text widgets render first either way), adjust the index in the failing test so it points at the correct element. If no failures, no adjustment is needed.

> Do NOT commit yet.

---

## Task 5: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Overwrite `CLAUDE.md` with the text-only version**

Replace the entire contents of `CLAUDE.md` with:

```markdown
## Project

Streamlit app for translating text across 67 languages using mlx-community/tiny-aya-global-8bit-mlx (8-bit quantized multilingual model) with local MLX inference on Apple Silicon.

## Stack

- Python 3.12+, uv for project management
- Streamlit (UI), mlx-lm (inference on Apple Silicon)
- python-dotenv for configuration

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for pure functions
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
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- All utility buttons use Material Icons via the `icon` parameter and `help=` for hover tooltips
- Swap button (`:material/swap_horiz:`, tertiary) flips languages via `st.session_state` and moves output into input
- Translate button uses `type="primary"` with a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled) with a controls row (Translate, clear, copy, download)
- Copy button uses `subprocess` + `/usr/bin/pbcopy` for plain-text clipboard with `st.toast("Translation copied")` on success
- Download button uses `st.download_button` to save translation as `translation.txt`
- `translate_text` builds a chat prompt, formats it with `tokenizer.apply_chat_template`, creates a sampler via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- Model loaded once via `@st.cache_resource` using `mlx_lm.load`; runs on Apple Silicon only
- Config loaded from `.env` via python-dotenv with sensible defaults
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` level (not `streamlit_app`) because AppTest runs scripts via `exec()`
- License: CC-BY-NC (non-commercial use only)
```

- [ ] **Step 2: Sanity-check the file no longer mentions removed components**

Run:
```bash
grep -n "document\|docx\|pptx\|xlsx\|pdf\|pymupdf\|openpyxl\|tab\|Documents\|_replace_paragraph_text\|translate_document" CLAUDE.md || echo "clean"
```

Expected: `clean`.

> Do NOT commit yet.

---

## Task 6: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Overwrite `README.md` with the text-only version**

Replace the entire contents of `README.md` with:

```markdown
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
```

- [ ] **Step 2: Sanity-check the file no longer mentions documents**

Run:
```bash
grep -n "document\|docx\|pptx\|xlsx\|pdf\|test_document" README.md || echo "clean"
```

Expected: `clean`.

> Do NOT commit yet.

---

## Task 7: Full verification

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

Run:
```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v
```

Expected: all tests pass. (`test_document.py` is gone, so do not list it.)

- [ ] **Step 2: Run ruff lint**

Run:
```bash
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Run ruff format check**

Run:
```bash
uv run ruff format --check .
```

Expected: no files would be reformatted. If ruff reports changes, run `uv run ruff format .` and re-run the check.

- [ ] **Step 4: Run type check**

Run:
```bash
uv run ty check streamlit_app.py
```

Expected: no type errors.

- [ ] **Step 5: Smoke-start the app (manual, optional but recommended)**

Run:
```bash
uv run streamlit run streamlit_app.py
```

Expected: app loads without errors; a single page appears with the language bar, side-by-side input/output, and controls row (no tab header). Stop the server with Ctrl+C after confirming. This step requires a human to inspect the browser; skip it if running non-interactively and rely on the test suite.

- [ ] **Step 6: Review final staged/unstaged changes**

Run:
```bash
git status
git diff --stat
```

Expected modified files: `streamlit_app.py`, `test_streamlit_ui.py`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`.
Expected deleted files: `document.py`, `test_document.py`.

---

## Task 8: Commit

**Files:** all of the above.

- [ ] **Step 1: Stage all changes**

Run:
```bash
git add streamlit_app.py test_streamlit_ui.py pyproject.toml uv.lock CLAUDE.md README.md document.py test_document.py
```

(`git add` on the deleted files records the deletions.)

- [ ] **Step 2: Verify the staged change set is correct**

Run:
```bash
git status
```

Expected: only the eight files above appear, each as either `modified` or `deleted`. Nothing else should be staged.

- [ ] **Step 3: Create the commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
refactor: remove documents tab and doc processing

Simplifies the app to text-only translation. Deletes document.py and
test_document.py, drops the Documents tab from the UI, flattens the
Text-tab body (no more single-tab st.tabs shell), and removes
python-docx, python-pptx, openpyxl, and pymupdf from dependencies.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds.

- [ ] **Step 4: Confirm the commit landed**

Run:
```bash
git log -1 --stat
```

Expected: one commit on top of `main` with the eight files listed (two deletions, six modifications).
