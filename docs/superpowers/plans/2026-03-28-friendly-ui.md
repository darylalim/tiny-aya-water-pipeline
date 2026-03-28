# Friendly UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app friendlier for non-technical colleagues by adding region-filtered language dropdowns and updating copy throughout.

**Architecture:** Add a `LANGUAGE_GROUPS` dictionary alongside the existing `LANGUAGES` list. Each language selectbox gets a region radio filter above it that controls which languages appear. All labels, placeholders, and warnings are updated to use friendlier language.

**Tech Stack:** Python, Streamlit (native components only), pytest

---

### Task 1: Add LANGUAGE_GROUPS data structure

**Files:**
- Modify: `streamlit_app.py:27-73` (after LANGUAGES list)
- Modify: `test_streamlit_app.py:7-17` (imports), `test_streamlit_app.py:103-115` (after LANGUAGES tests)

- [ ] **Step 1: Add LANGUAGE_GROUPS import to test file**

In `test_streamlit_app.py`, add `LANGUAGE_GROUPS` to the imports:

```python
from streamlit_app import (
    LANGUAGE_GROUPS,
    LANGUAGES,
    build_summarization_prompt,
    build_translation_prompt,
    clean_model_output,
    detect_device,
    get_summary_config,
    select_dtype,
    summarize_text,
    translate_text,
)
```

- [ ] **Step 2: Write failing tests for LANGUAGE_GROUPS**

In `test_streamlit_app.py`, add after the `test_languages_list_contains_japanese` test (line 115):

```python
# -- LANGUAGE_GROUPS -----------------------------------------------------------


def test_language_groups_has_two_regions() -> None:
    assert set(LANGUAGE_GROUPS.keys()) == {"European", "Asia-Pacific"}


def test_language_groups_european_has_31_languages() -> None:
    assert len(LANGUAGE_GROUPS["European"]) == 31


def test_language_groups_asia_pacific_has_12_languages() -> None:
    assert len(LANGUAGE_GROUPS["Asia-Pacific"]) == 12


def test_language_groups_covers_all_languages() -> None:
    all_grouped = LANGUAGE_GROUPS["European"] + LANGUAGE_GROUPS["Asia-Pacific"]
    assert sorted(all_grouped) == sorted(LANGUAGES)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_app.py::test_language_groups_has_two_regions -v`
Expected: FAIL with `ImportError` (LANGUAGE_GROUPS not defined)

- [ ] **Step 4: Add LANGUAGE_GROUPS to streamlit_app.py**

In `streamlit_app.py`, add after the `LANGUAGES` list (after line 73):

```python
LANGUAGE_GROUPS: dict[str, list[str]] = {
    "European": [
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
    ],
    "Asia-Pacific": [
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
    ],
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -k "language_groups" -v`
Expected: 4 PASSED

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All tests PASS (no existing tests affected)

- [ ] **Step 7: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add LANGUAGE_GROUPS data structure with region grouping"
```

---

### Task 2: Update translate tab with region filtering and friendly copy

**Files:**
- Modify: `streamlit_app.py:258-294` (translate tab)
- Modify: `test_streamlit_ui.py:125-219` (translate tab tests)

- [ ] **Step 1: Write new failing tests for region radio defaults**

In `test_streamlit_ui.py`, add after `test_translate_tab_button_exists` (line 162):

```python
def test_translate_tab_source_region_default(app: AppTest) -> None:
    tab = app.tabs[0]
    assert tab.radio[0].value == "European"


def test_translate_tab_target_region_default(app: AppTest) -> None:
    tab = app.tabs[0]
    assert tab.radio[1].value == "European"
```

- [ ] **Step 2: Write new failing tests for region filtering**

In `test_streamlit_ui.py`, add after the tests from step 1:

```python
def test_translate_source_region_filters_languages(app: AppTest) -> None:
    """Switching source region to Asia-Pacific shows Asia-Pacific languages."""
    app.tabs[0].radio[0].set_value("Asia-Pacific")
    _rerun_with_mocks(app)

    assert app.tabs[0].selectbox[0].value == "Chinese"


def test_translate_target_region_filters_languages(app: AppTest) -> None:
    """Switching target region to Asia-Pacific shows Asia-Pacific languages."""
    app.tabs[0].radio[1].set_value("Asia-Pacific")
    _rerun_with_mocks(app)

    assert app.tabs[0].selectbox[1].value == "Chinese"
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_translate_tab_source_region_default -v`
Expected: FAIL with `IndexError` (no radio widgets in translate tab yet)

- [ ] **Step 4: Implement translate tab changes**

In `streamlit_app.py`, replace the entire translate tab block (lines 258–294) with:

```python
with translate_tab:
    st.markdown("**① Pick your languages**")
    col1, col2 = st.columns(2)
    with col1:
        source_region = st.radio(
            "Region",
            list(LANGUAGE_GROUPS.keys()),
            horizontal=True,
            key="source_region",
        )
        source_languages = LANGUAGE_GROUPS[source_region]
        source_lang = st.selectbox("Source Language", source_languages)
    with col2:
        target_region = st.radio(
            "Region",
            list(LANGUAGE_GROUPS.keys()),
            horizontal=True,
            key="target_region",
        )
        target_languages = LANGUAGE_GROUPS[target_region]
        target_lang = st.selectbox(
            "Target Language",
            target_languages,
            index=target_languages.index("French")
            if "French" in target_languages
            else 0,
        )

    st.divider()
    st.markdown("**② Type or paste your text**")
    translate_input = st.text_area(
        "Text to translate",
        placeholder="e.g. The weather is nice today",
        height=150,
    )

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
            st.divider()
            st.markdown("**③ Translation**")
            st.success(result)
```

- [ ] **Step 5: Update existing translate tab label tests**

In `test_streamlit_ui.py`, update `test_translate_tab_has_choose_languages_label` (line 125–128):

```python
def test_translate_tab_has_choose_languages_label(app: AppTest) -> None:
    tab = app.tabs[0]
    markdown_values = [m.value for m in tab.markdown]
    assert any("① Pick your languages" in v for v in markdown_values)
```

Update `test_translate_tab_has_enter_text_label` (line 131–134):

```python
def test_translate_tab_has_enter_text_label(app: AppTest) -> None:
    tab = app.tabs[0]
    markdown_values = [m.value for m in tab.markdown]
    assert any("② Type or paste your text" in v for v in markdown_values)
```

- [ ] **Step 6: Update existing translate tab interaction tests**

Update `test_translate_success_shows_result_label` (line 175–179):

```python
def test_translate_success_shows_result_label() -> None:
    """After a successful translation the '③ Translation' label is shown."""
    at = _run_inference_test(tab_index=0, input_text="Hello", decode_result="Bonjour")
    markdown_values = [m.value for m in at.tabs[0].markdown]
    assert any("③ Translation" in v for v in markdown_values)
```

Update `test_translate_empty_text_shows_warning` (line 182–189):

```python
def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    """Clicking Translate with an empty text area shows a warning."""
    app.tabs[0].button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[0]
    warning_values = [w.value for w in tab.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)
```

Update `test_translate_same_language_shows_warning` (line 192–203):

```python
def test_translate_same_language_shows_warning(app: AppTest) -> None:
    """Translating when source == target language shows a warning."""
    tab = app.tabs[0]
    tab.selectbox[1].set_value("English")
    tab.text_area[0].set_value("Hello")
    tab.button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[0]
    warning_values = [w.value for w in tab.warning]
    assert any("two different languages" in str(v) for v in warning_values)
```

Update `test_translate_change_target_language` (line 214–219) — Japanese is in Asia-Pacific so the old test won't work with European default; switch to a European language:

```python
def test_translate_change_target_language(app: AppTest) -> None:
    """Changing the target language selectbox updates its value."""
    app.tabs[0].selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.tabs[0].selectbox[1].value == "Spanish"
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All tests PASS

- [ ] **Step 8: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 9: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: add region filtering to translate tab with friendly copy"
```

---

### Task 3: Update summarize tab with region filtering and friendly copy

**Files:**
- Modify: `streamlit_app.py:296-330` (summarize tab — line numbers from original; adjust based on Task 2 changes)
- Modify: `test_streamlit_ui.py:225-308` (summarize tab tests)

- [ ] **Step 1: Write new failing tests for region radio**

In `test_streamlit_ui.py`, add after `test_summarize_tab_button_exists`:

```python
def test_summarize_tab_output_region_default(app: AppTest) -> None:
    tab = app.tabs[1]
    assert tab.radio[1].value == "European"


def test_summarize_output_region_filters_languages(app: AppTest) -> None:
    """Switching output region to Asia-Pacific shows Asia-Pacific languages."""
    app.tabs[1].radio[1].set_value("Asia-Pacific")
    _rerun_with_mocks(app)

    assert app.tabs[1].selectbox[0].value == "Chinese"
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest test_streamlit_ui.py::test_summarize_tab_output_region_default -v`
Expected: FAIL with `IndexError` (only one radio in summarize tab currently)

- [ ] **Step 3: Implement summarize tab changes**

In `streamlit_app.py`, replace the entire summarize tab block with:

```python
with summarize_tab:
    st.markdown("**① Pick your options**")
    col1, col2 = st.columns(2)
    with col1:
        summary_length = st.radio(
            "Summary Length", ["Short", "Medium", "Long"], horizontal=True
        )
    with col2:
        output_region = st.radio(
            "Region",
            list(LANGUAGE_GROUPS.keys()),
            horizontal=True,
            key="output_region",
        )
        output_languages = LANGUAGE_GROUPS[output_region]
        output_lang = st.selectbox("Output Language", output_languages)

    st.divider()
    st.markdown("**② Type or paste your text**")
    summarize_input = st.text_area(
        "Text to summarize",
        placeholder="e.g. Paste an article, email, or paragraph here",
        height=150,
    )

    if st.button("Summarize", disabled=not model_loaded):
        if not summarize_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Summarizing..."):
                result = summarize_text(
                    summarize_input,
                    output_lang,
                    summary_length,
                    model,
                    tokenizer,
                )
            st.divider()
            st.markdown("**③ Summary**")
            st.success(result)
```

- [ ] **Step 4: Update existing summarize tab tests**

Update `test_summarize_tab_has_choose_options_label` (line 225–228):

```python
def test_summarize_tab_has_choose_options_label(app: AppTest) -> None:
    tab = app.tabs[1]
    markdown_values = [m.value for m in tab.markdown]
    assert any("① Pick your options" in v for v in markdown_values)
```

Update `test_summarize_tab_has_enter_text_label` (line 231–234):

```python
def test_summarize_tab_has_enter_text_label(app: AppTest) -> None:
    tab = app.tabs[1]
    markdown_values = [m.value for m in tab.markdown]
    assert any("② Type or paste your text" in v for v in markdown_values)
```

Update `test_summarize_tab_text_area_placeholder` (line 252–256):

```python
def test_summarize_tab_text_area_placeholder(app: AppTest) -> None:
    tab = app.tabs[1]
    text_area = tab.text_area[0]
    assert text_area.placeholder == "e.g. Paste an article, email, or paragraph here"
```

Update `test_summarize_success_shows_result_label` (line 276–282):

```python
def test_summarize_success_shows_result_label() -> None:
    """After a successful summarize the '③ Summary' label is shown."""
    at = _run_inference_test(
        tab_index=1, input_text="Some long text.", decode_result="A brief summary."
    )
    markdown_values = [m.value for m in at.tabs[1].markdown]
    assert any("③ Summary" in v for v in markdown_values)
```

Update `test_summarize_empty_text_shows_warning` (line 285–292):

```python
def test_summarize_empty_text_shows_warning(app: AppTest) -> None:
    """Clicking Summarize with an empty text area shows a warning."""
    app.tabs[1].button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[1]
    warning_values = [w.value for w in tab.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All tests PASS

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: add region filtering to summarize tab with friendly copy"
```

---

### Task 4: Update page header and caption

**Files:**
- Modify: `streamlit_app.py:232-247` (title, subtitle, caption — line numbers from original; adjust based on prior tasks)
- Modify: `test_streamlit_ui.py:112-119` (caption tests)

- [ ] **Step 1: Update caption test assertion**

In `test_streamlit_ui.py`, update `test_caption_contains_language_count` (line 117–119):

```python
def test_caption_contains_language_count(app: AppTest) -> None:
    captions = [c.value for c in app.caption]
    assert any("43 European and Asia-Pacific languages" in c for c in captions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_streamlit_ui.py::test_caption_contains_language_count -v`
Expected: FAIL (current caption says "43 languages", not "43 European and Asia-Pacific languages")

- [ ] **Step 3: Update subtitle in streamlit_app.py**

Replace the `st.markdown(...)` subtitle call with:

```python
st.markdown(
    "Translate and summarize text across 43 languages — all running privately on "
    "your computer."
)
```

- [ ] **Step 4: Update caption in streamlit_app.py**

Replace the `st.caption(...)` call with:

```python
    st.caption(
        f"Powered by [tiny-aya-water]({model_url}) · Supports 43 European and "
        f"Asia-Pacific languages"
    )
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All tests PASS

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py test_streamlit_ui.py
git commit -m "feat: update page header and caption with friendlier copy"
```
