# Simplified UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the Streamlit UI by removing the sidebar, batch processing, and settings controls, replacing the radio task toggle with tabs.

**Architecture:** Single-file refactor of `streamlit_app.py`. All pure functions and tests are unchanged. The UI section (lines 260-479) is rewritten: sidebar removed, title/subtitle moved above model loading, radio replaced with `st.tabs`, batch UI deleted, temperature/max_tokens removed from call sites.

**Tech Stack:** Streamlit, Python

**Spec:** `docs/superpowers/specs/2026-03-26-simplified-ui-design.md`

---

### Task 1: Remove sidebar and reorder header

**Files:**
- Modify: `streamlit_app.py:260-294`

- [ ] **Step 1: Delete the sidebar section**

Delete lines 260-274 (the entire sidebar block):

```python
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
    "Model: [CohereLabs/tiny-aya-water]"
    "(https://huggingface.co/CohereLabs/tiny-aya-water)  \n"
    "License: CC-BY-NC (non-commercial)"
)
```

- [ ] **Step 2: Rewrite the model loading and header section**

Replace the current model loading block (lines 276-294) with title/subtitle first, then model loading, then caption:

```python
# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water")
st.markdown(
    "Translate and summarize across 43 European and Asia-Pacific languages using "
    "[CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) "
    "running locally."
)

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model, device, dtype = load_model()
    st.caption(
        f"Model: [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) "
        f"| Device: {device} | Dtype: {dtype} | License: CC-BY-NC"
    )
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests pass (pure functions unchanged)

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: remove sidebar and reorder header above model loading"
```

---

### Task 2: Replace task selector with tabs and implement Translate tab

**Files:**
- Modify: `streamlit_app.py:296-393`

- [ ] **Step 1: Replace radio toggle and translate section with tab-based Translate tab**

Delete everything from the `task = st.radio(...)` line through the end of the Translate section (current lines 296-393). Replace with:

```python
# -- Tabs ---------------------------------------------------------------------

translate_tab, summarize_tab = st.tabs(["Translate", "Summarize"])

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

    input_text = st.text_area("Text to translate", height=150)

    if st.button("Translate", disabled=not model_loaded):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        elif source_lang == target_lang:
            st.warning("Source and target language are the same.")
        else:
            with st.spinner("Translating..."):
                result = translate_text(
                    input_text,
                    source_lang,
                    target_lang,
                    model,
                    tokenizer,
                )
            st.text_area("Translation", value=result, height=150, disabled=True)
```

Note: `temperature` and `max_tokens` arguments are removed from the `translate_text` call — function defaults apply.

- [ ] **Step 2: Run tests to verify nothing broke**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: replace radio toggle with tabs and simplify Translate tab"
```

---

### Task 3: Implement Summarize tab

**Files:**
- Modify: `streamlit_app.py` (immediately after the Translate tab `with` block)

- [ ] **Step 1: Replace the Summarize section with tab-based Summarize tab**

Delete the entire `else:` block (current Summarize mode, lines 395-479). Replace with:

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

    input_text = st.text_area("Text to summarize", height=150)

    if st.button("Summarize", disabled=not model_loaded):
        if not input_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                result = summarize_text(
                    input_text,
                    output_lang,
                    summary_length,
                    model,
                    tokenizer,
                )
            st.text_area("Summary", value=result, height=150, disabled=True)
```

Note: `temperature` and `max_tokens` arguments are removed from the `summarize_text` call — function defaults apply.

- [ ] **Step 2: Run tests to verify nothing broke**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: simplify Summarize tab and remove batch UI"
```

---

### Task 4: Lint, format, and type check

**Files:**
- Modify: `streamlit_app.py` (if fixes needed)

- [ ] **Step 1: Run linter**

Run: `uv run ruff check --fix .`
Expected: No errors (or auto-fixed)

- [ ] **Step 2: Run formatter**

Run: `uv run ruff format .`
Expected: File formatted (or already formatted)

- [ ] **Step 3: Run type checker**

Run: `uv run ty check streamlit_app.py`
Expected: No errors

- [ ] **Step 4: Run full test suite one final time**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit any formatting/lint fixes**

```bash
git add streamlit_app.py
git commit -m "chore: lint and format after UI simplification"
```

(Skip if no changes were made.)
