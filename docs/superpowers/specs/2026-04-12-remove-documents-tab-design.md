# Remove Documents tab, focus on text-only translation

**Date:** 2026-04-12
**Status:** Approved

## Goal

Simplify the app to a single-purpose text translator. Delete the Documents tab, the `document.py` module, its tests, and the four document-processing dependencies. Flatten the UI so there is no single-tab `st.tabs` shell.

## Motivation

The app currently has two tabs (Text, Documents). The Documents tab adds a `document.py` module, a dedicated test file, four runtime dependencies (`python-docx`, `python-pptx`, `openpyxl`, `pymupdf`), and duplicated session state / language-bar UI. Removing it collapses the codebase to one concern and trims the dependency surface. Reverting later is a single `git revert` if needed.

## Scope

**In scope**
- Delete the Documents tab UI and all doc-only session state + callbacks in `streamlit_app.py`.
- Flatten the Text tab body so there is no `st.tabs` wrapper.
- Delete `document.py` and `test_document.py`.
- Drop `openpyxl`, `pymupdf`, `python-docx`, `python-pptx` from `pyproject.toml` and regenerate `uv.lock`.
- Update `CLAUDE.md`, `README.md`, and `pyproject.toml` description to reflect text-only scope.
- Update `test_streamlit_ui.py` to remove doc-tab tests.

**Out of scope**
- No behavior changes to text translation (`translate_text`, prompt, sampler, cleaning).
- No changes to the `LANGUAGES` list, configuration, or `.env.example`.
- No visual changes to the Text tab beyond unwrapping it from `st.tabs`.
- No renaming of the project, package, or repo.

## Changes by file

### `streamlit_app.py`

- Remove the `tab_text, tab_docs = st.tabs(["Text", "Documents"])` call and the entire `with tab_docs:` block (currently L387–504).
- Replace `with tab_text:` with direct top-level statements; unindent its body one level.
- Delete doc-only session-state defaults: `doc_source_lang`, `doc_target_lang`, `doc_translated_bytes`, `doc_translated_filename`, `_do_translate_doc`.
- Delete doc-only callbacks: `swap_doc_languages`, `request_translate_doc`.
- Leave untouched: imports, config block, `LANGUAGES`, all pure functions (`build_translation_prompt`, `clean_model_output`, `translate_text`), `load_model`, `st.title`, model-loading block, and the Text-tab callbacks (`request_translate`, `swap_languages`, `clear_input`).

### `document.py`

Delete the file.

### `test_document.py`

Delete the file.

### `pyproject.toml`

- In `[project].dependencies`, remove `openpyxl>=3.1.0`, `pymupdf>=1.25.0`, `python-docx>=1.1.0`, `python-pptx>=1.0.0`. Keep `mlx-lm`, `python-dotenv`, `streamlit`.
- Update `[project].description` from "Translate text and documents across 67 languages with mlx-community/tiny-aya-global-8bit-mlx on Apple Silicon" to a text-only phrasing, e.g. "Translate text across 67 languages with mlx-community/tiny-aya-global-8bit-mlx on Apple Silicon".
- Run `uv sync` to regenerate `uv.lock`.

### `CLAUDE.md`

- **Project** section: drop "and documents"; drop the list of supported file formats.
- **Stack** section: remove `python-docx`, `python-pptx`, `openpyxl`, `pymupdf` bullets.
- **Structure** section: remove the `document.py` and `test_document.py` lines.
- **Commands** section: update the test command to `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`.
- **Conventions** section: remove items that reference the Documents tab, `doc_source_lang`/`doc_target_lang`, `_do_translate_doc`, `st.tabs`, per-format `extract_segments_*` / `rebuild_document_*` pairs, `translate_document`, `_replace_paragraph_text`, supported file types, file-uploader behavior, and the "Documents tab:" bullet. Adjust the remaining Text-tab bullets to no longer imply a tabbed layout (e.g., "Text tab: side-by-side…" → "Side-by-side…").

### `README.md`

- Update the one-line description under the title to drop "and documents".
- **Features** section: remove the "Document translation for .docx, .pptx, .xlsx, and .pdf…" bullet.
- **Development** section: update the test command to `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v`.

### `test_streamlit_ui.py`

- Remove the entire "Documents tab" test section (currently `test_tabs_exist`, `test_doc_translate_button_exists`, `test_doc_translate_button_disabled_when_no_file`, `test_doc_translate_button_disabled_when_model_fails`, `test_doc_language_defaults`, `test_doc_language_independent_from_text_tab`, `test_text_language_independent_from_doc_tab`, `test_doc_swap_flips_languages`, `test_doc_file_uploader_exists`).
- The Text-tab tests should not need index changes: with tabs removed, the first two `selectbox` elements remain the Text source/target, and the two `text_area` elements remain input/output. Verify by running the suite; if any Text-tab test breaks due to ordering, adjust in place.

## Verification

1. `uv sync` completes cleanly with the reduced dependency set.
2. `uv run pytest test_streamlit_app.py test_streamlit_ui.py -v` — all tests pass.
3. `uv run ruff check .` and `uv run ruff format --check .` — clean.
4. `uv run ty check streamlit_app.py` — clean.
5. Manual smoke test: `uv run streamlit run streamlit_app.py`, then exercise translate, swap, clear, copy, download on the Text page; confirm no tab bar is shown and no document-related UI remains.

## Commit shape

A single commit: `refactor: remove documents tab and doc processing`. This makes `git revert` a one-shot recovery if we want the feature back later.

## Risks

- **Test index drift in `test_streamlit_ui.py`.** AppTest's element indexing is position-based; removing the Documents tab should not affect Text-tab indices because Text widgets render first regardless, but this must be confirmed by running the suite.
- **Lockfile churn.** Removing four deps will trim `uv.lock` noticeably. The resulting lockfile diff is expected and belongs in the same commit.
- **Stale documentation.** `CLAUDE.md` and `README.md` both reference documents in several places; each reference must be updated so the docs match the code. The implementation plan should enumerate each reference rather than relying on a blanket edit.
