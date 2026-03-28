# Guided Flow UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Streamlit UI into a guided step-by-step flow with numbered steps, dividers, and visual hierarchy for non-technical users.

**Architecture:** UI-only changes to `streamlit_app.py` lines 230-315. Each tab gets three numbered steps with `st.markdown` bold labels and `st.divider()` separators. Output switches from disabled `st.text_area` to `st.success`. No pure function or test changes.

**Tech Stack:** Streamlit (`st.markdown`, `st.divider`, `st.success`)

---

### Task 1: Simplify Model Caption

**Files:**
- Modify: `streamlit_app.py:244-249`

- [ ] **Step 1: Update the model caption**

Replace the technical caption with a user-friendly version:

```python
    model_url = "https://huggingface.co/CohereLabs/tiny-aya-water"
    st.caption(
        f"Powered by [tiny-aya-water]({model_url}) · 43 languages"
    )
```

The old code to replace (lines 244-249):

```python
    model_url = "https://huggingface.co/CohereLabs/tiny-aya-water"
    st.caption(
        f"Model: [CohereLabs/tiny-aya-water]({model_url}) "
        f"| Device: {device} | Dtype: {dtype} "
        f"| License: CC-BY-NC"
    )
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check streamlit_app.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: simplify model caption for non-technical users"
```

---

### Task 2: Restructure Translate Tab with Guided Flow

**Files:**
- Modify: `streamlit_app.py:260-287`

- [ ] **Step 1: Replace translate tab contents**

Replace the current translate tab code (lines 260-287):

```python
with translate_tab:
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Source Language", LANGUAGES, index=LANGUAGES.index("English")
        )
    with col2:
        target_lang = st.selectbox(
            "Target Language", LANGUAGES, index=LANGUAGES.index("French")
        )

    translate_input = st.text_area("Text to translate", height=150)

    if st.button("Translate", disabled=not model_loaded):
        if not translate_input.strip():
            st.warning("Please enter some text to translate.")
        elif source_lang == target_lang:
            st.warning("Source and target language are the same.")
        else:
            with st.spinner("Translating..."):
                result = translate_text(
                    translate_input,
                    source_lang,
                    target_lang,
                    model,
                    tokenizer,
                )
            st.text_area("Translation", value=result, height=150, disabled=True)
```

With the guided flow version:

```python
with translate_tab:
    st.markdown("**① Choose languages**")
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Source Language", LANGUAGES, index=LANGUAGES.index("English")
        )
    with col2:
        target_lang = st.selectbox(
            "Target Language", LANGUAGES, index=LANGUAGES.index("French")
        )

    st.divider()
    st.markdown("**② Enter text**")
    translate_input = st.text_area(
        "Text to translate",
        placeholder="e.g. The weather is nice today",
        height=150,
    )

    if st.button("Translate", disabled=not model_loaded):
        if not translate_input.strip():
            st.warning("Please enter some text to translate.")
        elif source_lang == target_lang:
            st.warning("Source and target language are the same.")
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
            st.markdown("**③ Result**")
            st.success(result)
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check streamlit_app.py`
Expected: No errors

- [ ] **Step 3: Run tests to verify no regressions**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All 42 tests pass

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: restructure Translate tab with guided flow steps"
```

---

### Task 3: Restructure Summarize Tab with Guided Flow

**Files:**
- Modify: `streamlit_app.py:289-314` (line numbers after Task 2 edits — find the `with summarize_tab:` block)

- [ ] **Step 1: Replace summarize tab contents**

Replace the current summarize tab code:

```python
with summarize_tab:
    col1, col2 = st.columns(2)
    with col1:
        summary_length = st.radio(
            "Summary Length", ["Short", "Medium", "Long"], horizontal=True
        )
    with col2:
        output_lang = st.selectbox(
            "Output Language", LANGUAGES, index=LANGUAGES.index("English")
        )

    summarize_input = st.text_area("Text to summarize", height=150)

    if st.button("Summarize", disabled=not model_loaded):
        if not summarize_input.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                result = summarize_text(
                    summarize_input,
                    output_lang,
                    summary_length,
                    model,
                    tokenizer,
                )
            st.text_area("Summary", value=result, height=150, disabled=True)
```

With the guided flow version:

```python
with summarize_tab:
    st.markdown("**① Choose options**")
    col1, col2 = st.columns(2)
    with col1:
        summary_length = st.radio(
            "Summary Length", ["Short", "Medium", "Long"], horizontal=True
        )
    with col2:
        output_lang = st.selectbox(
            "Output Language", LANGUAGES, index=LANGUAGES.index("English")
        )

    st.divider()
    st.markdown("**② Enter text**")
    summarize_input = st.text_area(
        "Text to summarize",
        placeholder="Paste an article, paragraph, or any text to summarize...",
        height=150,
    )

    if st.button("Summarize", disabled=not model_loaded):
        if not summarize_input.strip():
            st.warning("Please enter some text to summarize.")
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
            st.markdown("**③ Result**")
            st.success(result)
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check streamlit_app.py`
Expected: No errors

- [ ] **Step 3: Run tests to verify no regressions**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All 42 tests pass

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: restructure Summarize tab with guided flow steps"
```

---

### Task 4: Remove Unused Variables from load_model Destructuring

**Files:**
- Modify: `streamlit_app.py:243` and `streamlit_app.py:253`

After Task 1, the `device` and `dtype` variables are no longer used in the success branch. Clean up the destructuring.

- [ ] **Step 1: Update the success branch destructuring**

Change line 243 from:

```python
        tokenizer, model, device, dtype = load_model()
```

To:

```python
        tokenizer, model, _device, _dtype = load_model()
```

- [ ] **Step 2: Update the failure branch to match**

Change line 253 from:

```python
    tokenizer, model, device, dtype = None, None, None, None
```

To:

```python
    tokenizer, model = None, None
```

- [ ] **Step 3: Run lint**

Run: `uv run ruff check streamlit_app.py`
Expected: No errors. If ruff flags `_device`/`_dtype` as unused, that is expected — the underscore prefix suppresses the warning.

- [ ] **Step 4: Run tests to verify no regressions**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All 42 tests pass

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: remove unused device/dtype variables from UI code"
```

---

### Task 5: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In the Conventions section, update the bullet about the model caption. Change:

```
- Model loaded once via `@st.cache_resource` with `dtype` (not deprecated `torch_dtype`); inference runs under `torch.inference_mode()`
```

No change needed — this convention is about `load_model()` internals, not the caption. Review the rest of CLAUDE.md; if there is a bullet referencing the caption format, update it. Otherwise no CLAUDE.md change is needed.

- [ ] **Step 2: Update README.md if it references UI layout**

Read `README.md`. If it describes the UI layout (tabs, controls), update to mention the guided step flow. If it only describes how to run the app, no change is needed.

- [ ] **Step 3: Commit (only if changes were made)**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update docs for guided flow UI"
```
