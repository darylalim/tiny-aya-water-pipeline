# Guided Flow UI Design

## Goal

Improve visual clarity for non-technical users by restructuring each tab into a guided step-by-step flow with clear visual grouping, replacing the current flat layout where controls and text areas run together without hierarchy.

## Target Audience

Non-technical users who want to translate or summarize text quickly with minimal friction.

## Design Principles

- Clear top-to-bottom flow with numbered steps
- Visual separation between steps using dividers and bold labels
- Approachable language (no technical jargon on screen)
- Result only appears after action (no empty output box)
- All Streamlit-native components, no custom CSS

## Changes

### Header

- **Model caption** changes from `"Model: [CohereLabs/tiny-aya-water](...) | Device: {device} | Dtype: {dtype} | License: CC-BY-NC"` to `"Powered by [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) · 43 languages"`
- Removes device, dtype, and license from the visible caption — these are technical details not useful to non-technical users
- Title and subtitle remain unchanged

### Translate Tab

Three steps separated by `st.divider()`, each introduced by `st.markdown("**① ...**")`:

**① Choose languages**
- Two `st.columns`: source language selectbox (default: English) and target language selectbox (default: French)
- Language list unchanged (43 languages)

**② Enter text**
- `st.text_area("Text to translate", placeholder="e.g. The weather is nice today", height=150)`
- `st.button("Translate")` directly below, disabled when model not loaded

**③ Result**
- Only rendered after translation runs (not a permanent empty box)
- Output displayed via `st.success(result)` instead of `st.text_area(..., disabled=True)`

Validation unchanged: `st.warning` on empty text, `st.warning` when source equals target language.

### Summarize Tab

Same three-step pattern:

**① Choose options**
- Two `st.columns`: summary length radio (Short/Medium/Long, horizontal) on the left, output language selectbox (default: English) on the right

**② Enter text**
- `st.text_area("Text to summarize", placeholder="Paste an article, paragraph, or any text to summarize...", height=150)`
- `st.button("Summarize")` directly below, disabled when model not loaded

**③ Result**
- Only rendered after summarization runs
- Output displayed via `st.success(result)`

Validation unchanged: `st.warning` on empty text.

### Unchanged

- Page title (`st.title("Tiny Aya Water")`) and subtitle (`st.markdown(...)`)
- Tab structure: `st.tabs(["Translate", "Summarize"])`
- All pure functions: `translate_text`, `summarize_text`, `_generate`, `build_translation_prompt`, `build_summarization_prompt`, `clean_model_output`, `detect_device`, `select_dtype`, `get_summary_config`
- All config variables: `MODEL_ID`, `DEVICE`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`
- `@st.cache_resource` model loading with spinner and error handling
- `LANGUAGES` list
- Spinners during inference
- Temperature and max tokens use function defaults (configurable via `.env`)

## Execution Order

1. `st.title("Tiny Aya Water")`
2. `st.markdown(...)` — description paragraph
3. Model loading via `load_model()` inside `st.spinner`
4. On success: `st.caption("Powered by [tiny-aya-water](...) · 43 languages")`
5. On failure: `st.error(...)` (no caption)
6. `st.tabs(["Translate", "Summarize"])` — tab contents follow

## Testing Impact

- All existing pure function tests remain unchanged
- No new tests needed — changes are UI-only (step labels, dividers, placeholder text, `st.success` output)
