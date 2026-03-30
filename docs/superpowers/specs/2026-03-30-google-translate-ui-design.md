# Google Translate-Inspired UI Design

## Goal

Update the Tiny Aya Water Translate UI to match Google Translate conventions: matched panel heights, character count, clear button, copy button, and cleaner placeholder behavior.

## Changes

### 1. Switch output from `st.code()` to `st.text_area(disabled=True)`

- **Current:** `st.code(st.session_state.translate_output, language=None)` — auto-sized, monospace, built-in copy button
- **New:** `st.text_area("Output", value=st.session_state.translate_output, disabled=True, height=300, placeholder="Translation", label_visibility="collapsed")`
- Enables fixed height matching, placeholder support, and consistent styling with the input panel
- Trade-off: loses built-in copy button (replaced by change #7)

### 2. Increase input text area height

- **Current:** `height=200`
- **New:** `height=300` to comfortably accommodate up to 5,000 characters
- Output panel uses the same `height=300` so both match

### 3. Remove input placeholder text

- **Current:** `placeholder="Type or paste your text here..."`
- **New:** Remove the `placeholder` parameter entirely

### 4. Add "Translation" placeholder to output panel

- **New:** `placeholder="Translation"` on the disabled output text area
- Shows "Translation" in muted text when the output is empty

### 5. Add character count below input

- Display `"{count} / 5,000"` below the input text area using `st.caption()`
- Right-aligned, below the input panel
- Count updates reactively based on `translate_input` length

### 6. Add clear (✕) button below input

- `st.button("✕")` placed below the input panel, left side
- On click: clears `st.session_state.translate_input` and `st.session_state.translate_output`
- Disabled when the input is empty (`disabled=not translate_input.strip()`)
- Uses `type="tertiary"` for minimal visual weight

### 7. Add copy (⧉) button below output

- `st.button("⧉")` placed below the output panel, right-aligned
- On click: copies `st.session_state.translate_output` to clipboard
- Clipboard access requires a small JS snippet injected via `st.html()` since Streamlit has no native clipboard API
- Icon only, no text label

## Layout Structure

```
[Title: Tiny Aya Water Translate]

[Source Lang ▾]  [⇄]  [Target Lang ▾]

[Input text_area, 300px]    [Output text_area disabled, 300px]
[✕ clear]    [25 / 5,000]  [                          ⧉ copy]

[Translate button]
```

- Below-input row: ✕ on the left, character count on the right (using `st.columns`)
- Below-output row: ⧉ on the right (using `st.columns`)

## Implementation Notes

- The ✕ and ⧉ buttons use narrow columns to position them at the edges
- Clipboard JS: `st.html()` injects a `<script>` tag that calls `navigator.clipboard.writeText()` — triggered by setting a session state flag that conditionally renders the script
- The `swap_languages` callback continues to work as before: swaps languages and moves output into input, clearing the output
- Character count is a pure computation from `len(translate_input)`, formatted with comma separator

## Files Affected

- `streamlit_app.py` — UI section (lines ~250–283): replace `st.code` with `st.text_area`, add character count, clear button, copy button
- `test_streamlit_ui.py` — update tests referencing `app.code` to use `app.text_area`, add tests for new UI elements (clear button, character count, copy button)

## Out of Scope

- No CSS injection for panel styling (uses native Streamlit components)
- No auto-translate on typing (keeps explicit Translate button)
- No language chips or detect-language feature
- No changes to pure functions, model loading, or session state logic (except new clear/copy callbacks)
