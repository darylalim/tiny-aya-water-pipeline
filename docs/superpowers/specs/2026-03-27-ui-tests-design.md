# UI Tests Design

## Goal

Add Streamlit AppTest-based UI tests to verify the guided flow layout, widget defaults, validation messages, and interaction behavior introduced by the guided flow UI changes.

## Approach

Use `AppTest.from_file("streamlit_app.py")` with `unittest.mock.patch` to mock model loading and inference functions. Tests run the actual app file for real confidence.

## Test File

- **New file:** `test_streamlit_ui.py` — separate from existing `test_streamlit_app.py` (pure function tests)
- **Framework:** `streamlit.testing.v1.AppTest` (available in Streamlit 1.55.0)

## Mocking Strategy

- Patch `streamlit_app.load_model` to return `(mock_tokenizer, mock_model, "cpu", torch.float32)` — avoids downloading the 7GB model
- Patch `streamlit_app.translate_text` to return `"Bonjour"`
- Patch `streamlit_app.summarize_text` to return `"A brief summary."`
- Shared pytest fixture creates the patched `AppTest` instance, calls `.run()`, and returns it

## Tests

### Caption

- `st.caption` contains `"Powered by"` and `"43 languages"`

### Translate Tab — Structure

- Step labels `"① Choose languages"` and `"② Enter text"` present in markdown elements
- Divider exists between steps
- Two selectboxes with defaults: English (source), French (target)
- Text area with placeholder `"e.g. The weather is nice today"`
- Translate button exists and is enabled

### Translate Tab — Successful Translation

- Set text area value, click Translate
- `st.success` appears with `"Bonjour"`
- Step label `"③ Result"` appears in markdown

### Translate Tab — Validation

- Empty text + click Translate → `st.warning` contains "Please enter some text"
- Same source/target language + text + click Translate → `st.warning` contains "same"

### Translate Tab — Language Selection

- Change source to Spanish, target to Japanese → selectbox values update correctly

### Summarize Tab — Structure

- Step labels `"① Choose options"` and `"② Enter text"` present in markdown elements
- Divider exists between steps
- Radio with Short/Medium/Long options
- Selectbox with default English (output language)
- Text area with placeholder `"Paste an article, paragraph, or any text to summarize..."`
- Summarize button exists and is enabled

### Summarize Tab — Successful Summarization

- Set text area value, click Summarize
- `st.success` appears with `"A brief summary."`
- Step label `"③ Result"` appears in markdown

### Summarize Tab — Validation

- Empty text + click Summarize → `st.warning` contains "Please enter some text"

### Summarize Tab — Options

- Change radio to "Long" → value updates
- Change output language to French → selectbox value updates

## Testing Impact

- No changes to `streamlit_app.py` or `test_streamlit_app.py`
- New file `test_streamlit_ui.py` only
- Run command: `uv run pytest test_streamlit_ui.py -v`
