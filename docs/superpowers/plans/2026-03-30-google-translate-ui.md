# Google Translate-Inspired UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update text panels, add character count, clear button, and copy button to match Google Translate conventions.

**Architecture:** All changes are in the UI section of `streamlit_app.py` (lines 250–283). Output switches from `st.code()` to `st.text_area(disabled=True)` with matching height. New sub-columns below each panel hold the clear/count and copy controls. Clipboard uses `st.html()` JS injection.

**Tech Stack:** Streamlit (text_area, button, caption, columns, html), json (for JS string escaping)

---

### Task 1: Update text panels — replace st.code, adjust heights, update placeholders

**Files:**
- Modify: `test_streamlit_ui.py:94-99,105-114,117-145,151-156,166-168`
- Modify: `streamlit_app.py:251-263`

- [ ] **Step 1: Write failing tests for new panel behavior**

Update existing tests and add new ones in `test_streamlit_ui.py`. Replace all `app.code` references with `app.text_area[1]`, update placeholder tests, and add output placeholder test.

Replace these test functions:

```python
def test_input_text_area_has_no_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == ""


def test_output_uses_text_area(app: AppTest) -> None:
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[1].placeholder == "Translation"
```

Update `test_translate_success_shows_result`:

```python
def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    assert at.text_area[1].value == "Bonjour"
```

Update `test_swap_moves_output_to_input` — change the last assertion:

```python
    # Output should be cleared
    assert at.text_area[1].value == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_input_text_area_has_no_placeholder test_streamlit_ui.py::test_output_uses_text_area test_streamlit_ui.py::test_output_text_area_placeholder test_streamlit_ui.py::test_translate_success_shows_result test_streamlit_ui.py::test_swap_moves_output_to_input -v`

Expected: FAIL — `test_input_text_area_has_no_placeholder` (old function name still exists or placeholder mismatch), `test_output_uses_text_area` (only 1 text_area exists), `test_output_text_area_placeholder` (no second text_area), `test_translate_success_shows_result` (no `text_area[1]`), `test_swap_moves_output_to_input` (no `text_area[1]`)

- [ ] **Step 3: Implement panel changes in streamlit_app.py**

Replace the side-by-side text panels section (lines 251–263):

```python
# -- Side-by-side text panels -------------------------------------------------

col_input, col_output = st.columns(2)
with col_input:
    translate_input = st.text_area(
        "Input",
        height=300,
        key="translate_input",
        label_visibility="collapsed",
    )
with col_output:
    st.text_area(
        "Output",
        height=300,
        placeholder="Translation",
        disabled=True,
        key="translate_output",
        label_visibility="collapsed",
    )
```

Key changes from the current code:
- Input: `height` 200 → 300, `placeholder` removed
- Output: `st.code()` → `st.text_area(disabled=True)` with `height=300`, `placeholder="Translation"`, `key="translate_output"` (binds to existing session state)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: switch output to text_area, match panel heights, update placeholders"
```

---

### Task 2: Add character count below input

**Files:**
- Modify: `test_streamlit_ui.py`
- Modify: `streamlit_app.py:251-262`

- [ ] **Step 1: Write failing test for character count**

Add to `test_streamlit_ui.py`:

```python
def test_character_count_shown(app: AppTest) -> None:
    assert any("0 / 5,000" in c.value for c in app.caption)


def test_character_count_updates(app: AppTest) -> None:
    app.text_area[0].set_value("Hello")
    _rerun_with_mocks(app)

    assert any("5 / 5,000" in c.value for c in app.caption)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_character_count_shown test_streamlit_ui.py::test_character_count_updates -v`

Expected: FAIL — no caption elements exist yet

- [ ] **Step 3: Implement character count**

In `streamlit_app.py`, add a `st.caption` below the input text area inside `col_input`. Update the `with col_input:` block:

```python
with col_input:
    translate_input = st.text_area(
        "Input",
        height=300,
        key="translate_input",
        label_visibility="collapsed",
    )
    st.caption(f"{len(translate_input):,} / 5,000")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: add character count below input text area"
```

---

### Task 3: Add clear (✕) button below input

**Files:**
- Modify: `test_streamlit_ui.py`
- Modify: `streamlit_app.py:214-222,251-263`

- [ ] **Step 1: Write failing tests for clear button**

Add to `test_streamlit_ui.py`:

```python
def test_clear_button_exists(app: AppTest) -> None:
    assert app.button("✕") is not None


def test_clear_button_disabled_when_input_empty(app: AppTest) -> None:
    assert app.button("✕").disabled


def test_clear_button_clears_input_and_output() -> None:
    """After translating, clicking ✕ should clear both panels."""
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

        # Click clear
        at.button("✕").click()
        at.run(timeout=60)

    assert at.text_area[0].value == ""
    assert at.text_area[1].value == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_clear_button_exists test_streamlit_ui.py::test_clear_button_disabled_when_input_empty test_streamlit_ui.py::test_clear_button_clears_input_and_output -v`

Expected: FAIL — no "✕" button exists

- [ ] **Step 3: Implement clear button**

Add the `clear_input` callback in `streamlit_app.py` after `swap_languages`:

```python
def clear_input() -> None:
    """Clear the input and output text."""
    st.session_state.translate_input = ""
    st.session_state.translate_output = ""
```

Update the `with col_input:` block to add the clear button and rearrange with the character count using sub-columns:

```python
with col_input:
    translate_input = st.text_area(
        "Input",
        height=300,
        key="translate_input",
        label_visibility="collapsed",
    )
    sub_clear, sub_count = st.columns(2)
    with sub_clear:
        st.button(
            "✕",
            on_click=clear_input,
            disabled=not translate_input.strip(),
            type="tertiary",
        )
    with sub_count:
        st.caption(f"{len(translate_input):,} / 5,000")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: add clear button below input text area"
```

---

### Task 4: Add copy (⧉) button with clipboard JS

**Files:**
- Modify: `test_streamlit_ui.py`
- Modify: `streamlit_app.py:1-5,263-270`

- [ ] **Step 1: Write failing test for copy button**

Add to `test_streamlit_ui.py`:

```python
def test_copy_button_exists(app: AppTest) -> None:
    assert app.button("⧉") is not None


def test_copy_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.button("⧉").disabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_copy_button_exists test_streamlit_ui.py::test_copy_button_disabled_when_output_empty -v`

Expected: FAIL — no "⧉" button exists

- [ ] **Step 3: Implement copy button**

Add `import json` to the top of `streamlit_app.py` with the other stdlib imports:

```python
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
```

Update the `with col_output:` block to add the copy button below the output text area:

```python
with col_output:
    st.text_area(
        "Output",
        height=300,
        placeholder="Translation",
        disabled=True,
        key="translate_output",
        label_visibility="collapsed",
    )
    _, sub_copy = st.columns([5, 1])
    with sub_copy:
        if st.button("⧉", type="tertiary", disabled=not st.session_state.translate_output.strip()):
            js_text = json.dumps(st.session_state.translate_output)
            st.html(f"<script>navigator.clipboard.writeText({js_text});</script>")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_ui.py -v`

Expected: All tests PASS

- [ ] **Step 5: Run full test suite and lints**

```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v
uv run ruff check --fix .
uv run ruff format .
```

Expected: All tests PASS, no lint errors

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: add copy button with clipboard JS below output panel"
```
