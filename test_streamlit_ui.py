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
    assert app.button("⇄") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("⇄").click()
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
        at.button("⇄").click()
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


# -- Character count -----------------------------------------------------------


def test_character_count_shown(app: AppTest) -> None:
    assert any("0 / 5,000" in c.value for c in app.caption)


def test_character_count_updates(app: AppTest) -> None:
    app.text_area[0].set_value("Hello")
    _rerun_with_mocks(app)

    assert any("5 / 5,000" in c.value for c in app.caption)


# -- Clear button --------------------------------------------------------------


def test_clear_button_exists(app: AppTest) -> None:
    assert app.button("✕") is not None


def test_clear_button_disabled_when_input_empty(app: AppTest) -> None:
    assert app.button("✕").disabled


def test_clear_button_clears_input_and_output() -> None:
    """After translating, clicking ✕ should clear both panels."""
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
        at.button("✕").click()
        at.run(timeout=60)

    assert at.text_area[0].value == ""
    assert at.text_area[1].value == ""


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
