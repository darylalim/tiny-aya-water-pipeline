# Document Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add file upload and translation for .docx, .pdf, .pptx, and .xlsx documents with formatting preserved, accessed via a "Documents" tab alongside the existing "Text" tab.

**Architecture:** New `document.py` module with per-format extract/rebuild functions and a `translate_document` coordinator that accepts a translation callback. UI adds `st.tabs` with the existing text UI in the first tab and a new document upload/translate/download flow in the second tab. Each tab has independent language selection.

**Tech Stack:** python-docx, python-pptx, openpyxl, pymupdf (fitz), Streamlit tabs

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `document.py` | Per-format extract/rebuild functions + `translate_document` coordinator |
| Create | `test_document.py` | Unit tests for all document processing functions |
| Modify | `pyproject.toml` | Add python-docx, python-pptx, openpyxl, pymupdf |
| Modify | `streamlit_app.py:191-378` | Wrap in tabs, add Documents tab UI |
| Modify | `test_streamlit_ui.py` | Add Documents tab UI tests |
| Modify | `CLAUDE.md` | Update conventions for document translation |

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:7-13`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

```toml
dependencies = [
    "accelerate>=1.13.0",
    "openpyxl>=3.1.0",
    "pymupdf>=1.25.0",
    "python-docx>=1.1.0",
    "python-dotenv>=1.2.2",
    "python-pptx>=1.0.0",
    "streamlit>=1.55.0",
    "torch>=2.11.0",
    "transformers>=5.3.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: All packages install successfully.

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import docx; import pptx; import openpyxl; import fitz; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add document processing dependencies"
```

---

### Task 2: DOCX Extract and Rebuild

**Files:**
- Create: `document.py`
- Create: `test_document.py`

- [ ] **Step 1: Write test helpers and extract test**

Create `test_document.py`:

```python
from __future__ import annotations

import io

from docx import Document

from document import extract_segments_docx


def _make_docx(paragraphs: list[str]) -> bytes:
    """Create a minimal DOCX with the given paragraph texts."""
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_docx_with_table(
    paragraphs: list[str], table_rows: list[list[str]]
) -> bytes:
    """Create a DOCX with body paragraphs and a table."""
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
    for i, row_data in enumerate(table_rows):
        for j, cell_text in enumerate(row_data):
            table.rows[i].cells[j].text = cell_text
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -- extract_segments_docx -----------------------------------------------------


def test_extract_docx_returns_paragraph_texts() -> None:
    file_bytes = _make_docx(["Hello", "World"])
    segments = extract_segments_docx(file_bytes)
    assert segments == ["Hello", "World"]


def test_extract_docx_includes_empty_paragraphs() -> None:
    file_bytes = _make_docx(["Hello", "", "World"])
    segments = extract_segments_docx(file_bytes)
    assert segments == ["Hello", "", "World"]


def test_extract_docx_includes_table_text() -> None:
    file_bytes = _make_docx_with_table(["Intro"], [["Cell A", "Cell B"]])
    segments = extract_segments_docx(file_bytes)
    assert "Intro" in segments
    assert "Cell A" in segments
    assert "Cell B" in segments
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_extract_docx_returns_paragraph_texts -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'document'`

- [ ] **Step 3: Create document.py with DOCX extract**

Create `document.py`:

```python
from __future__ import annotations

import io
from typing import Any


# -- Helpers -------------------------------------------------------------------


def _replace_paragraph_text(para: Any, text: str) -> None:
    """Replace all text in a paragraph, preserving the first run's formatting."""
    if para.runs:
        para.runs[0].text = text
        for run in para.runs[1:]:
            run.text = ""
    elif text.strip():
        para.add_run().text = text


# -- DOCX ---------------------------------------------------------------------


def _iter_docx_paragraphs(doc: Any) -> list[Any]:
    """Return all paragraphs in document body and tables, deduped."""
    seen: set[int] = set()
    paragraphs: list[Any] = []
    for para in doc.paragraphs:
        pid = id(para._element)
        if pid not in seen:
            seen.add(pid)
            paragraphs.append(para)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    pid = id(para._element)
                    if pid not in seen:
                        seen.add(pid)
                        paragraphs.append(para)
    return paragraphs


def extract_segments_docx(file_bytes: bytes) -> list[str]:
    """Extract translatable text segments from a DOCX file."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    return [para.text for para in _iter_docx_paragraphs(doc)]
```

- [ ] **Step 4: Run extract tests to verify pass**

Run: `uv run pytest test_document.py -k "extract_docx" -v`
Expected: 3 tests PASS

- [ ] **Step 5: Write failing rebuild test**

Add to `test_document.py`:

```python
from document import extract_segments_docx, rebuild_document_docx


def test_rebuild_docx_replaces_text() -> None:
    file_bytes = _make_docx(["Hello", "World"])
    rebuilt = rebuild_document_docx(file_bytes, ["Bonjour", "Monde"])
    segments = extract_segments_docx(rebuilt)
    assert segments == ["Bonjour", "Monde"]


def test_rebuild_docx_round_trip() -> None:
    original = ["Hello", "World"]
    file_bytes = _make_docx(original)
    segments = extract_segments_docx(file_bytes)
    rebuilt = rebuild_document_docx(file_bytes, segments)
    result = extract_segments_docx(rebuilt)
    assert result == original
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_rebuild_docx_replaces_text -v`
Expected: FAIL — `ImportError: cannot import name 'rebuild_document_docx'`

- [ ] **Step 7: Implement rebuild_document_docx**

Add to `document.py` after `extract_segments_docx`:

```python
def rebuild_document_docx(file_bytes: bytes, translations: list[str]) -> bytes:
    """Rebuild a DOCX file with translated text replacing original segments."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    for i, para in enumerate(_iter_docx_paragraphs(doc)):
        if i >= len(translations):
            break
        _replace_paragraph_text(para, translations[i])
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
```

- [ ] **Step 8: Run all DOCX tests to verify pass**

Run: `uv run pytest test_document.py -k "docx" -v`
Expected: 5 tests PASS

- [ ] **Step 9: Commit**

```bash
git add document.py test_document.py
git commit -m "feat: add DOCX extract and rebuild with tests"
```

---

### Task 3: PPTX Extract and Rebuild

**Files:**
- Modify: `document.py`
- Modify: `test_document.py`

- [ ] **Step 1: Write PPTX test helpers and extract test**

Add to `test_document.py`:

```python
from pptx import Presentation
from pptx.util import Inches

from document import extract_segments_pptx


def _make_pptx(texts: list[str]) -> bytes:
    """Create a minimal PPTX with one slide and one textbox per text."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    for i, text in enumerate(texts):
        txBox = slide.shapes.add_textbox(
            Inches(1), Inches(0.5 + i * 1.5), Inches(5), Inches(1)
        )
        txBox.text_frame.paragraphs[0].text = text
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_extract_pptx_returns_paragraph_texts() -> None:
    file_bytes = _make_pptx(["Hello", "World"])
    segments = extract_segments_pptx(file_bytes)
    assert segments == ["Hello", "World"]


def test_extract_pptx_includes_empty_paragraphs() -> None:
    file_bytes = _make_pptx(["Hello", "", "World"])
    segments = extract_segments_pptx(file_bytes)
    assert segments == ["Hello", "", "World"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_extract_pptx_returns_paragraph_texts -v`
Expected: FAIL — `ImportError: cannot import name 'extract_segments_pptx'`

- [ ] **Step 3: Implement PPTX extract**

Add to `document.py`:

```python
# -- PPTX ---------------------------------------------------------------------


def _iter_pptx_paragraphs(prs: Any) -> list[Any]:
    """Return all paragraphs across all slides, shapes, and tables."""
    paragraphs: list[Any] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    paragraphs.append(para)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        for para in cell.text_frame.paragraphs:
                            paragraphs.append(para)
    return paragraphs


def extract_segments_pptx(file_bytes: bytes) -> list[str]:
    """Extract translatable text segments from a PPTX file."""
    from pptx import Presentation

    prs = Presentation(io.BytesIO(file_bytes))
    return [para.text for para in _iter_pptx_paragraphs(prs)]
```

- [ ] **Step 4: Run extract tests to verify pass**

Run: `uv run pytest test_document.py -k "extract_pptx" -v`
Expected: 2 tests PASS

- [ ] **Step 5: Write failing rebuild test**

Add to `test_document.py`:

```python
from document import extract_segments_pptx, rebuild_document_pptx


def test_rebuild_pptx_replaces_text() -> None:
    file_bytes = _make_pptx(["Hello", "World"])
    rebuilt = rebuild_document_pptx(file_bytes, ["Bonjour", "Monde"])
    segments = extract_segments_pptx(rebuilt)
    assert segments == ["Bonjour", "Monde"]


def test_rebuild_pptx_round_trip() -> None:
    original = ["Hello", "World"]
    file_bytes = _make_pptx(original)
    segments = extract_segments_pptx(file_bytes)
    rebuilt = rebuild_document_pptx(file_bytes, segments)
    result = extract_segments_pptx(rebuilt)
    assert result == original
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_rebuild_pptx_replaces_text -v`
Expected: FAIL — `ImportError: cannot import name 'rebuild_document_pptx'`

- [ ] **Step 7: Implement rebuild_document_pptx**

Add to `document.py` after `extract_segments_pptx`:

```python
def rebuild_document_pptx(file_bytes: bytes, translations: list[str]) -> bytes:
    """Rebuild a PPTX file with translated text replacing original segments."""
    from pptx import Presentation

    prs = Presentation(io.BytesIO(file_bytes))
    for i, para in enumerate(_iter_pptx_paragraphs(prs)):
        if i >= len(translations):
            break
        _replace_paragraph_text(para, translations[i])
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
```

- [ ] **Step 8: Run all PPTX tests to verify pass**

Run: `uv run pytest test_document.py -k "pptx" -v`
Expected: 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add document.py test_document.py
git commit -m "feat: add PPTX extract and rebuild with tests"
```

---

### Task 4: XLSX Extract and Rebuild

**Files:**
- Modify: `document.py`
- Modify: `test_document.py`

- [ ] **Step 1: Write XLSX test helper and extract test**

Add to `test_document.py`:

```python
from openpyxl import Workbook

from document import extract_segments_xlsx


def _make_xlsx(rows: list[list[str]]) -> bytes:
    """Create a minimal XLSX with the given rows of string values."""
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_extract_xlsx_returns_cell_texts() -> None:
    file_bytes = _make_xlsx([["Hello", "World"], ["Foo", "Bar"]])
    segments = extract_segments_xlsx(file_bytes)
    assert segments == ["Hello", "World", "Foo", "Bar"]


def test_extract_xlsx_skips_non_string_cells() -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(["Hello", 42, None, "World"])
    buf = io.BytesIO()
    wb.save(buf)
    segments = extract_segments_xlsx(buf.getvalue())
    assert segments == ["Hello", "World"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_extract_xlsx_returns_cell_texts -v`
Expected: FAIL — `ImportError: cannot import name 'extract_segments_xlsx'`

- [ ] **Step 3: Implement XLSX extract**

Add to `document.py`:

```python
# -- XLSX ---------------------------------------------------------------------


def extract_segments_xlsx(file_bytes: bytes) -> list[str]:
    """Extract translatable text segments from an XLSX file."""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(file_bytes))
    segments: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    segments.append(cell.value)
    return segments
```

- [ ] **Step 4: Run extract tests to verify pass**

Run: `uv run pytest test_document.py -k "extract_xlsx" -v`
Expected: 2 tests PASS

- [ ] **Step 5: Write failing rebuild test**

Add to `test_document.py`:

```python
from document import extract_segments_xlsx, rebuild_document_xlsx


def test_rebuild_xlsx_replaces_text() -> None:
    file_bytes = _make_xlsx([["Hello", "World"]])
    rebuilt = rebuild_document_xlsx(file_bytes, ["Bonjour", "Monde"])
    segments = extract_segments_xlsx(rebuilt)
    assert segments == ["Bonjour", "Monde"]


def test_rebuild_xlsx_round_trip() -> None:
    original = ["Hello", "World", "Foo", "Bar"]
    file_bytes = _make_xlsx([["Hello", "World"], ["Foo", "Bar"]])
    segments = extract_segments_xlsx(file_bytes)
    rebuilt = rebuild_document_xlsx(file_bytes, segments)
    result = extract_segments_xlsx(rebuilt)
    assert result == original
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_rebuild_xlsx_replaces_text -v`
Expected: FAIL — `ImportError: cannot import name 'rebuild_document_xlsx'`

- [ ] **Step 7: Implement rebuild_document_xlsx**

Add to `document.py` after `extract_segments_xlsx`:

```python
def rebuild_document_xlsx(file_bytes: bytes, translations: list[str]) -> bytes:
    """Rebuild an XLSX file with translated text replacing original segments."""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(file_bytes))
    idx = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    if idx < len(translations):
                        cell.value = translations[idx]
                        idx += 1
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
```

- [ ] **Step 8: Run all XLSX tests to verify pass**

Run: `uv run pytest test_document.py -k "xlsx" -v`
Expected: 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add document.py test_document.py
git commit -m "feat: add XLSX extract and rebuild with tests"
```

---

### Task 5: PDF Extract and Rebuild

**Files:**
- Modify: `document.py`
- Modify: `test_document.py`

- [ ] **Step 1: Write PDF test helper and extract test**

Add to `test_document.py`:

```python
import fitz

from document import extract_segments_pdf


def _make_pdf(texts: list[str]) -> bytes:
    """Create a minimal PDF with one text block per string."""
    doc = fitz.open()
    page = doc.new_page()
    for i, text in enumerate(texts):
        rect = fitz.Rect(72, 72 + i * 50, 400, 72 + i * 50 + 40)
        page.insert_textbox(rect, text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_extract_pdf_returns_text_blocks() -> None:
    file_bytes = _make_pdf(["Hello", "World"])
    segments = extract_segments_pdf(file_bytes)
    assert "Hello" in segments
    assert "World" in segments


def test_extract_pdf_skips_empty_blocks() -> None:
    file_bytes = _make_pdf(["Hello", "World"])
    segments = extract_segments_pdf(file_bytes)
    assert all(s.strip() for s in segments)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_extract_pdf_returns_text_blocks -v`
Expected: FAIL — `ImportError: cannot import name 'extract_segments_pdf'`

- [ ] **Step 3: Implement PDF extract**

Add to `document.py`:

```python
# -- PDF ----------------------------------------------------------------------


def extract_segments_pdf(file_bytes: bytes) -> list[str]:
    """Extract translatable text segments from a PDF file."""
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    segments: list[str] = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]
            text = text.strip()
            if text:
                segments.append(text)
    doc.close()
    return segments
```

- [ ] **Step 4: Run extract tests to verify pass**

Run: `uv run pytest test_document.py -k "extract_pdf" -v`
Expected: 2 tests PASS

- [ ] **Step 5: Write failing rebuild test**

Add to `test_document.py`:

```python
from document import extract_segments_pdf, rebuild_document_pdf


def test_rebuild_pdf_inserts_translated_text() -> None:
    file_bytes = _make_pdf(["Hello", "World"])
    rebuilt = rebuild_document_pdf(file_bytes, ["Bonjour", "Monde"])
    segments = extract_segments_pdf(rebuilt)
    assert "Bonjour" in segments
    assert "Monde" in segments


def test_rebuild_pdf_preserves_segment_count() -> None:
    file_bytes = _make_pdf(["Hello", "World"])
    original_segments = extract_segments_pdf(file_bytes)
    rebuilt = rebuild_document_pdf(file_bytes, ["Bonjour", "Monde"])
    new_segments = extract_segments_pdf(rebuilt)
    assert len(new_segments) == len(original_segments)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_rebuild_pdf_inserts_translated_text -v`
Expected: FAIL — `ImportError: cannot import name 'rebuild_document_pdf'`

- [ ] **Step 7: Implement rebuild_document_pdf**

Add to `document.py` after `extract_segments_pdf`:

```python
def rebuild_document_pdf(file_bytes: bytes, translations: list[str]) -> bytes:
    """Rebuild a PDF with translated text replacing original text blocks.

    Best-effort: complex layouts may not reconstruct perfectly.
    """
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    idx = 0
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        insertions: list[tuple[fitz.Rect, str, float]] = []
        for block in blocks:
            if "lines" not in block:
                continue
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]
            text = text.strip()
            if not text:
                continue
            rect = fitz.Rect(block["bbox"])
            fontsize = block["lines"][0]["spans"][0]["size"]
            page.add_redact_annot(rect)
            if idx < len(translations):
                insertions.append((rect, translations[idx], fontsize))
                idx += 1
        page.apply_redactions()
        for rect, trans_text, fontsize in insertions:
            page.insert_textbox(rect, trans_text, fontsize=fontsize)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()
```

- [ ] **Step 8: Run all PDF tests to verify pass**

Run: `uv run pytest test_document.py -k "pdf" -v`
Expected: 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add document.py test_document.py
git commit -m "feat: add PDF extract and rebuild with tests"
```

---

### Task 6: translate_document Coordinator

**Files:**
- Modify: `document.py`
- Modify: `test_document.py`

- [ ] **Step 1: Write failing coordinator test**

Add to `test_document.py`:

```python
from document import translate_document


def test_translate_document_dispatches_to_docx() -> None:
    file_bytes = _make_docx(["Hello", "World"])
    translated = translate_document(
        file_bytes,
        "test.docx",
        translate_fn=lambda text: text.upper(),
    )
    segments = extract_segments_docx(translated)
    assert segments == ["HELLO", "WORLD"]


def test_translate_document_skips_empty_segments() -> None:
    file_bytes = _make_docx(["Hello", "", "World"])
    calls: list[str] = []

    def mock_translate(text: str) -> str:
        calls.append(text)
        return text.upper()

    translated = translate_document(file_bytes, "test.docx", translate_fn=mock_translate)
    segments = extract_segments_docx(translated)
    assert segments == ["HELLO", "", "WORLD"]
    assert calls == ["Hello", "World"]


def test_translate_document_unsupported_format() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported file format"):
        translate_document(b"data", "test.txt", translate_fn=lambda t: t)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_document.py::test_translate_document_dispatches_to_docx -v`
Expected: FAIL — `ImportError: cannot import name 'translate_document'`

- [ ] **Step 3: Implement translate_document**

Add to `document.py`:

```python
from pathlib import Path
from typing import Any, Callable


# -- Coordinator ---------------------------------------------------------------

HANDLERS: dict[str, tuple[Callable[[bytes], list[str]], Callable[[bytes, list[str]], bytes]]] = {
    ".docx": (extract_segments_docx, rebuild_document_docx),
    ".pdf": (extract_segments_pdf, rebuild_document_pdf),
    ".pptx": (extract_segments_pptx, rebuild_document_pptx),
    ".xlsx": (extract_segments_xlsx, rebuild_document_xlsx),
}


def translate_document(
    file_bytes: bytes,
    filename: str,
    translate_fn: Callable[[str], str],
) -> bytes:
    """Translate a document file, preserving formatting.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used to determine format from extension).
        translate_fn: A callable that takes a text string and returns its translation.

    Returns:
        The translated document as bytes in the same format.
    """
    ext = Path(filename).suffix.lower()
    if ext not in HANDLERS:
        msg = f"Unsupported file format: {ext}"
        raise ValueError(msg)
    extract_fn, rebuild_fn = HANDLERS[ext]
    segments = extract_fn(file_bytes)
    translations = [
        translate_fn(seg) if seg.strip() else seg
        for seg in segments
    ]
    return rebuild_fn(file_bytes, translations)
```

Note: Add `from pathlib import Path` to the top of `document.py` (new import). Add `Callable` to the existing `from typing import Any` import so it reads `from typing import Any, Callable`.

- [ ] **Step 4: Run all coordinator tests to verify pass**

Run: `uv run pytest test_document.py -k "translate_document" -v`
Expected: 3 tests PASS

- [ ] **Step 5: Run the full test_document.py suite**

Run: `uv run pytest test_document.py -v`
Expected: All tests PASS (approximately 20 tests)

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: No errors or all auto-fixed.

- [ ] **Step 7: Commit**

```bash
git add document.py test_document.py
git commit -m "feat: add translate_document coordinator with tests"
```

---

### Task 7: Restructure UI with Tabs

**Files:**
- Modify: `streamlit_app.py:191-378`

This task wraps the existing text translation UI inside the first tab and adds the tab structure. No new functionality — just restructuring.

- [ ] **Step 1: Add Documents tab session state defaults**

In `streamlit_app.py`, after the existing session state defaults block (after line 217), add:

```python
if "doc_source_lang" not in st.session_state:
    st.session_state.doc_source_lang = "English"
if "doc_target_lang" not in st.session_state:
    st.session_state.doc_target_lang = "French"
if "doc_translated_bytes" not in st.session_state:
    st.session_state.doc_translated_bytes = b""
if "doc_translated_filename" not in st.session_state:
    st.session_state.doc_translated_filename = ""
if "_do_translate_doc" not in st.session_state:
    st.session_state._do_translate_doc = False
```

- [ ] **Step 2: Add Documents tab callbacks**

After the existing `clear_input` callback (after line 238), add:

```python
def swap_doc_languages() -> None:
    """Swap source/target languages in the Documents tab."""
    st.session_state.doc_source_lang, st.session_state.doc_target_lang = (
        st.session_state.doc_target_lang,
        st.session_state.doc_source_lang,
    )


def request_translate_doc() -> None:
    """Flag that a document translation was requested."""
    st.session_state._do_translate_doc = True
```

- [ ] **Step 3: Add tabs and wrap Text tab UI**

Replace the code from the language bar comment (line 241) through the end of the file (line 378) with:

```python
# -- Tabs ---------------------------------------------------------------------

tab_text, tab_docs = st.tabs(["Text", "Documents"])

# -- Text tab -----------------------------------------------------------------

with tab_text:
    col_from, col_swap, col_to = st.columns(
        [10, 1, 10], vertical_alignment="center"
    )
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

    warning_slot = st.container()

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

    if st.session_state._do_translate:
        st.session_state._do_translate = False
        _current_input = st.session_state.translate_input
        if not _current_input.strip():
            warning_slot.warning("Please enter some text first.")
        elif st.session_state.source_lang == st.session_state.target_lang:
            warning_slot.warning("Please pick two different languages.")
        else:
            with warning_slot, st.spinner("Translating..."):
                result = translate_text(
                    _current_input,
                    st.session_state.source_lang,
                    st.session_state.target_lang,
                    model,
                    tokenizer,
                )
            st.session_state.translate_output = result
            st.rerun()
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All existing tests PASS. The Text tab's elements remain at the same indices since it's the first tab.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: wrap text translation UI in tabs"
```

---

### Task 8: Add Documents Tab UI

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Add the Documents tab UI**

In `streamlit_app.py`, after the `with tab_text:` block (after the `st.rerun()` line), add:

```python
# -- Documents tab ------------------------------------------------------------

with tab_docs:
    doc_col_from, doc_col_swap, doc_col_to = st.columns(
        [10, 1, 10], vertical_alignment="center"
    )
    with doc_col_from:
        st.selectbox(
            "From",
            LANGUAGES,
            key="doc_source_lang",
            label_visibility="collapsed",
        )
    with doc_col_swap:
        st.button(
            "",
            key="doc_swap",
            icon=":material/swap_horiz:",
            on_click=swap_doc_languages,
            use_container_width=True,
            type="tertiary",
            help="Swap languages",
        )
    with doc_col_to:
        st.selectbox(
            "To",
            LANGUAGES,
            key="doc_target_lang",
            label_visibility="collapsed",
        )

    doc_warning_slot = st.container()

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["docx", "pdf", "pptx", "xlsx"],
        label_visibility="collapsed",
    )

    _file_too_large = uploaded_file is not None and uploaded_file.size > 10 * 1024 * 1024
    if _file_too_large:
        doc_warning_slot.warning("File too large. Maximum size is 10 MB.")

    st.button(
        "Translate",
        key="TranslateDoc",
        on_click=request_translate_doc,
        disabled=not model_loaded or uploaded_file is None or _file_too_large,
        type="primary",
    )

    if st.session_state._do_translate_doc:
        st.session_state._do_translate_doc = False
        if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
            doc_warning_slot.warning("Please pick two different languages.")
        elif uploaded_file is None:
            doc_warning_slot.warning("Please upload a file first.")
        else:
            from document import translate_document

            def _translate_fn(text: str) -> str:
                return translate_text(
                    text,
                    st.session_state.doc_source_lang,
                    st.session_state.doc_target_lang,
                    model,
                    tokenizer,
                )

            with doc_warning_slot, st.spinner("Translating document..."):
                result_bytes = translate_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    translate_fn=_translate_fn,
                )
            target = st.session_state.doc_target_lang
            st.session_state.doc_translated_bytes = result_bytes
            st.session_state.doc_translated_filename = f"{target}_{uploaded_file.name}"
            st.rerun()

    if st.session_state.doc_translated_bytes:
        _mime_types = {
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pdf": "application/pdf",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        _fname = st.session_state.doc_translated_filename
        _ext = Path(_fname).suffix.lower()
        st.download_button(
            f"Download {_fname}",
            key="doc_download",
            data=st.session_state.doc_translated_bytes,
            file_name=_fname,
            mime=_mime_types.get(_ext, "application/octet-stream"),
            type="primary",
            icon=":material/download:",
        )
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`
Expected: All existing tests still PASS.

- [ ] **Step 3: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: No errors or all auto-fixed.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Documents tab with file upload and translation"
```

---

### Task 9: Documents Tab UI Tests

**Files:**
- Modify: `test_streamlit_ui.py`

- [ ] **Step 1: Write Documents tab UI tests**

Add to `test_streamlit_ui.py`:

```python
# -- Documents tab -------------------------------------------------------------


def test_tabs_exist(app: AppTest) -> None:
    assert len(app.tabs) > 0


def test_doc_translate_button_exists(app: AppTest) -> None:
    assert app.button("TranslateDoc") is not None


def test_doc_translate_button_disabled_when_no_file(app: AppTest) -> None:
    assert app.button("TranslateDoc").disabled


def test_doc_translate_button_disabled_when_model_fails() -> None:
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("download failed"),
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("TranslateDoc").disabled


def test_doc_language_defaults(app: AppTest) -> None:
    # Doc tab selectboxes are at index 2 and 3 (after text tab's two)
    assert app.selectbox[2].value == "English"
    assert app.selectbox[3].value == "French"


def test_doc_language_independent_from_text_tab(app: AppTest) -> None:
    # Change text tab language
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    # Doc tab languages unchanged
    assert app.selectbox[2].value == "English"
    assert app.selectbox[3].value == "French"


def test_doc_swap_flips_languages(app: AppTest) -> None:
    app.button("doc_swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[2].value == "French"
    assert app.selectbox[3].value == "English"


def test_doc_file_uploader_exists(app: AppTest) -> None:
    assert len(app.file_uploader) >= 1
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest test_streamlit_ui.py -k "doc_" -v`
Expected: All new tests PASS.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest test_streamlit_app.py test_streamlit_ui.py test_document.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add test_streamlit_ui.py
git commit -m "test: add Documents tab UI tests"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with document translation conventions**

Add the following to the relevant sections:

In **Structure**, add:
```
- `document.py` — document processing: per-format extract/rebuild + coordinator
- `test_document.py` — pytest unit tests for document processing
```

In **Conventions**, add:
```
- Document translation uses `document.py` with per-format `extract_segments_*` / `rebuild_document_*` pairs and a `translate_document` coordinator that accepts a `translate_fn` callback
- Supported formats: .docx (python-docx), .pptx (python-pptx), .xlsx (openpyxl), .pdf (pymupdf/fitz — best-effort)
- `_replace_paragraph_text` helper shared by DOCX and PPTX: replaces text preserving first run's formatting
- UI uses `st.tabs` with "Text" and "Documents" tabs; each tab has independent language selection (`source_lang`/`target_lang` vs `doc_source_lang`/`doc_target_lang`)
- Documents tab: file uploader (10 MB limit) → Translate button → download button; uses callback+flag pattern (`_do_translate_doc`)
- Downloaded document filename: `{target_lang}_{original_filename}`
```

- [ ] **Step 2: Lint and format all files**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for document translation feature"
```
