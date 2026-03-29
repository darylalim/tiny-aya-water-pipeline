# Side-by-Side Translate UI Design

Replace the tabbed Translate/Summarize layout with a single-page, side-by-side translation interface inspired by Google Translate. Add an interactive swap button for flipping languages and text.

## Goals

- Create a Google Translate-like side-by-side experience for input and output
- Add a swap button that flips languages and moves output into input
- Remove the Summarize tab to simplify the app to a single purpose
- Keep all pure functions intact for testability

## Layout

No tabs. Single page with three vertical sections:

1. **Language bar** — three `st.columns` in a 5:1:5 ratio:
   - Left: "From" `st.selectbox` (default: English)
   - Center: `st.button("⇄")` swap button
   - Right: "To" `st.selectbox` (default: French)

2. **Text panels** — two equal `st.columns`:
   - Left: `st.text_area` for input (placeholder: "Type or paste your text here...")
   - Right: `st.text_area` for output with `disabled=True` (placeholder empty, shows translation result)

3. **Translate button** — `st.button("Translate")` below the panels

## Subtitle

Change from "Translate and summarize text — running privately on your computer." to "Translate text — running privately on your computer."

## Swap Button Behavior

Clicking the swap button (⇄) triggers an `on_click` callback that:

1. Swaps `st.session_state` values for source and target language
2. Moves the current translation output into the input field via `st.session_state`
3. Clears the translation output from `st.session_state`

The swap button is always enabled regardless of model load state — it only manipulates UI state.

## Session State Keys

- `source_lang`: current source language (default: "English")
- `target_lang`: current target language (default: "French")
- `translate_input`: current input text
- `translate_output`: current translation result (empty string when no result)

## Validation

Unchanged from current behavior:

- Empty text: `st.warning("Please enter some text first.")`
- Same source/target language: `st.warning("Please pick two different languages.")`

## Removed Elements

| Element | Why |
|---|---|
| Summarize tab | App simplified to translate-only |
| `st.tabs` wrapper | No longer needed with single page |
| Static `→` arrow | Replaced by interactive `⇄` swap button |
| Summarize UI code | All `with summarize_tab:` block removed |

## Kept Elements

| Element | Why |
|---|---|
| All pure functions (including summarize) | Still importable and testable |
| `LANGUAGES` list | Used by translate selectboxes |
| Model loading (`load_model`, `@st.cache_resource`) | Unchanged |
| Device detection and config | Unchanged |
| `clean_model_output`, `_generate`, `translate_text` | Core translate pipeline |
| `build_summarization_prompt`, `summarize_text`, etc. | Kept as library functions, just not used by UI |

## Testing

### Unit tests (`test_streamlit_app.py`)

- All existing pure function tests remain unchanged (including summarize function tests)
- No new pure functions are introduced

### UI tests (`test_streamlit_ui.py`)

- Remove all Summarize tab tests
- Remove tab indexing (`app.tabs[0]`) — access widgets directly since there are no tabs
- Update Translate tests to use the new layout (two text areas instead of one, selectboxes accessed directly)
- Add new tests:
  - Swap button exists and has label "⇄"
  - Swap button flips source and target language values
  - Swap button moves output text into input and clears output
  - Output text area is disabled
  - Output text area shows translation result after clicking Translate
