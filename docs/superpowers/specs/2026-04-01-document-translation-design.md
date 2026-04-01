# Document Translation Feature — Design Spec

## Overview

Add file upload and translation support for .docx, .pdf, .pptx, and .xlsx documents. The output is a file of the same type with formatting preserved and text replaced with translations. Accessed via a "Documents" tab alongside the existing "Text" tab.

## Architecture

### Module: `document.py`

A new module containing pure functions for document processing. No Streamlit dependency.

**Per-format interface:**

```python
extract_segments(file_bytes: bytes) -> list[str]
rebuild_document(file_bytes: bytes, translations: list[str]) -> bytes
```

- `extract_segments` pulls translatable text segments from the document.
- `rebuild_document` takes the original file bytes and a list of translated strings (same length and order as `extract_segments` output), and returns the rebuilt file as bytes with translated text swapped in.
- Empty/whitespace-only segments are preserved as-is and not sent to the model.

**Format-specific behavior:**

| Format | Library | Extract | Rebuild |
|--------|---------|---------|---------|
| .docx | `python-docx` | Walk paragraphs → runs → collect text | Same traversal → replace run text |
| .pptx | `python-pptx` | Walk slides → shapes → text frames → paragraphs → runs | Same traversal → replace run text |
| .xlsx | `openpyxl` | Walk sheets → rows → cells with string values | Same traversal → replace cell values |
| .pdf | `pymupdf` | Extract text blocks per page | Redact original text → insert translated text in-place |

**Coordinator function:**

```python
translate_document(
    file_bytes: bytes,
    filename: str,
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> bytes
```

Determines format from file extension, calls `extract_segments`, translates each non-empty segment via the existing `translate_text()` (importable from `streamlit_app` since it's defined above the `import streamlit` line), calls `rebuild_document`, and returns the translated file as bytes.

### UI Changes: `streamlit_app.py`

**Tab layout:**

```
Title: "Tiny Aya Water Translate"
Model loading spinner
┌─────────────────────────────────────────────┐
│  [Text]  [Documents]                        │  ← st.tabs
├─────────────────────────────────────────────┤
│ Each tab contains:                          │
│   Language bar: [From] [swap] [To]          │
│   Warning slot                              │
│   Tab-specific content                      │
└─────────────────────────────────────────────┘
```

- Title and model loading remain above the tabs.
- Language bar renders **inside** each tab with **independent** language selections.
- Each tab has its own warning slot.
- Swap button in each tab operates on that tab's language state.

**Text tab:** The existing UI, unchanged, wrapped in `with tab_text:`.

**Documents tab session state:**

- `doc_source_lang` — source language (default: "English")
- `doc_target_lang` — target language (default: "French")
- `uploaded_file` — the uploaded file object
- `translated_file_bytes` — result bytes after translation
- `translated_filename` — output filename

**Documents tab UI flow:**

1. `st.file_uploader` accepting .docx, .pdf, .pptx, .xlsx with 10 MB limit.
2. "Translate" button (primary), disabled when no file uploaded or model not loaded.
3. Spinner while translating ("Translating document...").
4. Warnings in the tab's warning slot: no file uploaded, same language selected.
5. `st.download_button` appears after successful translation.
6. Output filename: `{target_lang}_{original_filename}` (e.g., `French_report.docx`).

## Dependencies

New production dependencies in `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `python-docx` | Read/write .docx |
| `python-pptx` | Read/write .pptx |
| `openpyxl` | Read/write .xlsx |
| `pymupdf` | Read/write .pdf |

No changes to dev dependencies.

## Testing

### `test_document.py` — Unit tests for document module

- `extract_segments` / `rebuild_document` tests for each format.
- Create minimal real files in-memory using the same libraries as test fixtures.
- Round-trip: extract → identity pass-through → rebuild → extract again → same text.
- Translated round-trip: extract → swap with known translations → rebuild → extract → translations present.
- Empty/whitespace segments preserved, not sent to model.
- `translate_document` coordinator: mock `translate_text`, verify correct format dispatch, returns bytes.

### `test_streamlit_ui.py` — UI tests for Documents tab

- Tab exists and is selectable.
- File uploader accepts only .docx, .pdf, .pptx, .xlsx.
- Translate button disabled when no file uploaded.
- Translate button disabled when model not loaded.
- Same-language warning appears.
- Download button appears after successful translation.
- Language bar in Documents tab is independent from Text tab.

## Constraints

- File size limit: 10 MB.
- PDF translation is best-effort — complex layouts (multi-column, overlapping text boxes) may not reconstruct perfectly.
- Translation is segment-by-segment — each text segment is translated independently without cross-segment context.
