from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> None:
    """Re-run the app with simple mocks (no generate chain needed)."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        app.run(timeout=60)


def _make_inference_mocks(decode_result: str) -> tuple[MagicMock, MagicMock]:
    """Return mocks configured for a successful _generate call."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    final_model = mock_model.to.return_value.eval.return_value
    final_model.device = torch.device("cpu")
    final_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = decode_result

    return mock_tokenizer, mock_model


def _run_inference_test(input_text: str, decode_result: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    mock_tokenizer, mock_model = _make_inference_mocks(decode_result)
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value(input_text)
        at.button("Translate").click()
        at.run(timeout=60)
    return at


# -- Language defaults ---------------------------------------------------------


def test_source_language_default(app: AppTest) -> None:
    assert app.selectbox[0].value == "English"


def test_target_language_default(app: AppTest) -> None:
    assert app.selectbox[1].value == "French"


# -- Swap button ---------------------------------------------------------------


def test_swap_button_exists(app: AppTest) -> None:
    assert app.button("swap") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_moves_output_to_input() -> None:
    """After translating, swap should move the output into the input field."""
    mock_tokenizer, mock_model = _make_inference_mocks("Bonjour")
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("Translate").click()
        at.run(timeout=60)

        # Swap
        at.button("swap").click()
        at.run(timeout=60)

    # Input should now contain the previous output
    assert at.text_area[0].value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


# -- Text panels ---------------------------------------------------------------


def test_input_text_area_has_no_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == ""


def test_output_uses_text_area(app: AppTest) -> None:
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[1].placeholder == "Translation"


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("Translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("Translate").disabled


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    assert at.text_area[1].value == "Bonjour"


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("Translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("Translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)


# -- Language switching --------------------------------------------------------


def test_change_source_language(app: AppTest) -> None:
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "Spanish"


def test_change_target_language(app: AppTest) -> None:
    app.selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[1].value == "Spanish"


# -- Input constraints ---------------------------------------------------------


def test_input_max_chars_enforced(app: AppTest) -> None:
    app.text_area[0].set_value("x" * 5001)
    _rerun_with_mocks(app)

    assert len(app.text_area[0].value) <= 5000


# -- Clear button --------------------------------------------------------------


def test_clear_button_exists(app: AppTest) -> None:
    assert app.button("clear") is not None


def test_clear_button_disabled_when_input_empty(app: AppTest) -> None:
    assert app.button("clear").disabled


def test_clear_button_enabled_when_input_has_text(app: AppTest) -> None:
    app.text_area[0].set_value("Hello")
    _rerun_with_mocks(app)

    assert not app.button("clear").disabled


def test_clear_button_clears_input_and_output() -> None:
    """After translating, clicking clear should clear both panels."""
    mock_tokenizer, mock_model = _make_inference_mocks("Bonjour")
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("Translate").click()
        at.run(timeout=60)

        # Click clear
        at.button("clear").click()
        at.run(timeout=60)

    assert at.text_area[0].value == ""
    assert at.text_area[1].value == ""


# -- Copy button ---------------------------------------------------------------


def test_copy_button_exists(app: AppTest) -> None:
    assert app.button("copy") is not None


def test_copy_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.button("copy").disabled


def test_copy_button_enabled_when_output_present() -> None:
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    assert not at.button("copy").disabled


def test_copy_button_click_no_errors() -> None:
    """Clicking copy with output present should not produce errors."""
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    at.button("copy").click()
    _rerun_with_mocks(at)

    assert not at.exception


def test_copy_button_shows_toast() -> None:
    """Clicking copy should show a 'Translation copied' toast."""
    at = _run_inference_test(input_text="Hello", decode_result="Bonjour")
    at.button("copy").click()
    _rerun_with_mocks(at)

    toast_values = [t.value for t in at.toast]
    assert any("Translation copied" in str(v) for v in toast_values)


# -- Output text area ----------------------------------------------------------


def test_output_text_area_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Model load failure --------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
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

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_model_load_failure_disables_translate_button() -> None:
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

    assert at.button("Translate").disabled


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


def test_text_language_independent_from_doc_tab(app: AppTest) -> None:
    # Change doc tab language
    app.selectbox[2].set_value("Spanish")
    _rerun_with_mocks(app)

    # Text tab languages unchanged
    assert app.selectbox[0].value == "English"
    assert app.selectbox[1].value == "French"


def test_doc_swap_flips_languages(app: AppTest) -> None:
    app.button("doc_swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[2].value == "French"
    assert app.selectbox[3].value == "English"


def test_doc_file_uploader_exists(app: AppTest) -> None:
    # AppTest in this Streamlit version doesn't expose a typed .file_uploader
    # accessor; fall back to the generic .get() which returns UnknownElement
    # entries for widgets not yet modelled by the testing API.
    assert len(app.get("file_uploader")) >= 1
