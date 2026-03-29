from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests.

    AppTest shares the Streamlit runtime within a process, so @st.cache_resource
    entries created by one test persist into the next.  Clearing before each test
    ensures every test starts with a clean model-loading state.
    """
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading.

    AppTest runs the script via exec() in the same process, so patches must
    target the upstream source of the objects the script imports.  load_model()
    does ``from transformers import AutoModelForCausalLM, AutoTokenizer``
    inside its body, so patching those names on the real ``transformers``
    module (approach 3) intercepts the call before any file I/O or network
    access occurs.
    """
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

    # Mirrors load_model(): model.to(device).eval()
    # If that chain changes, this mock chain must be updated to match.
    final_model = mock_model.to.return_value.eval.return_value
    final_model.device = torch.device("cpu")
    final_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = decode_result

    return mock_tokenizer, mock_model


def _run_inference_test(tab_index: int, input_text: str, decode_result: str) -> AppTest:
    """Build a fresh AppTest, enter text, click the action button, and return it."""
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
        at.tabs[tab_index].text_area[0].set_value(input_text)
        at.tabs[tab_index].button[0].click()
        at.run(timeout=60)
    return at


def test_translate_tab_source_language_default(app: AppTest) -> None:
    tab = app.tabs[0]
    source_select = tab.selectbox[0]
    assert source_select.value == "English"


def test_translate_tab_target_language_default(app: AppTest) -> None:
    tab = app.tabs[0]
    target_select = tab.selectbox[1]
    assert target_select.value == "French"


def test_translate_tab_text_area_placeholder(app: AppTest) -> None:
    tab = app.tabs[0]
    text_area = tab.text_area[0]
    assert text_area.placeholder == "Type or paste your text here..."


def test_translate_tab_button_exists(app: AppTest) -> None:
    tab = app.tabs[0]
    assert tab.button[0].label == "Translate"


def test_translate_success_shows_result() -> None:
    """Clicking Translate with valid input shows the translated text."""
    at = _run_inference_test(tab_index=0, input_text="Hello", decode_result="Bonjour")
    success_values = [s.value for s in at.tabs[0].success]
    assert any("Bonjour" in str(v) for v in success_values)


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    """Clicking Translate with an empty text area shows a warning."""
    app.tabs[0].button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[0]
    warning_values = [w.value for w in tab.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    """Translating when source == target language shows a warning."""
    tab = app.tabs[0]
    tab.selectbox[1].set_value("English")
    tab.text_area[0].set_value("Hello")
    tab.button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[0]
    warning_values = [w.value for w in tab.warning]
    assert any("two different languages" in str(v) for v in warning_values)


def test_translate_change_source_language(app: AppTest) -> None:
    """Changing the source language selectbox updates its value."""
    app.tabs[0].selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.tabs[0].selectbox[0].value == "Spanish"


def test_translate_change_target_language(app: AppTest) -> None:
    """Changing the target language selectbox updates its value."""
    app.tabs[0].selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.tabs[0].selectbox[1].value == "Spanish"


# -- Summarize tab: structure -------------------------------------------------


def test_summarize_tab_has_choose_options_label(app: AppTest) -> None:
    tab = app.tabs[1]
    markdown_values = [m.value for m in tab.markdown]
    assert any("① Pick your options" in v for v in markdown_values)


def test_summarize_tab_has_enter_text_label(app: AppTest) -> None:
    tab = app.tabs[1]
    markdown_values = [m.value for m in tab.markdown]
    assert any("② Type or paste your text" in v for v in markdown_values)


def test_summarize_tab_has_divider(app: AppTest) -> None:
    tab = app.tabs[1]
    assert len(tab.divider) >= 1


def test_summarize_tab_radio_default(app: AppTest) -> None:
    tab = app.tabs[1]
    assert tab.radio[0].value == "Short"


def test_summarize_tab_output_language_default(app: AppTest) -> None:
    tab = app.tabs[1]
    assert tab.selectbox[0].value == "English"


def test_summarize_tab_text_area_placeholder(app: AppTest) -> None:
    tab = app.tabs[1]
    text_area = tab.text_area[0]
    assert text_area.placeholder == "e.g. Paste an article, email, or paragraph here"


def test_summarize_tab_button_exists(app: AppTest) -> None:
    tab = app.tabs[1]
    assert tab.button[0].label == "Summarize"


def test_summarize_tab_output_region_default(app: AppTest) -> None:
    tab = app.tabs[1]
    assert tab.radio[1].value == "European"


def test_summarize_output_region_filters_languages(app: AppTest) -> None:
    """Switching output region to Asia-Pacific shows Asia-Pacific languages."""
    app.tabs[1].radio[1].set_value("Asia-Pacific")
    _rerun_with_mocks(app)

    assert app.tabs[1].selectbox[0].value == "Chinese"


# -- Summarize tab: interactions ----------------------------------------------


def test_summarize_success_shows_result() -> None:
    """Clicking Summarize with valid input shows the summarized text."""
    at = _run_inference_test(
        tab_index=1, input_text="Some long text.", decode_result="A brief summary."
    )
    success_values = [s.value for s in at.tabs[1].success]
    assert any("A brief summary." in str(v) for v in success_values)


def test_summarize_success_shows_result_label() -> None:
    """After a successful summarize the '③ Summary' label is shown."""
    at = _run_inference_test(
        tab_index=1, input_text="Some long text.", decode_result="A brief summary."
    )
    markdown_values = [m.value for m in at.tabs[1].markdown]
    assert any("③ Summary" in v for v in markdown_values)


def test_summarize_empty_text_shows_warning(app: AppTest) -> None:
    """Clicking Summarize with an empty text area shows a warning."""
    app.tabs[1].button[0].click()
    _rerun_with_mocks(app)

    tab = app.tabs[1]
    warning_values = [w.value for w in tab.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_summarize_change_radio_to_long(app: AppTest) -> None:
    """Changing the summary length radio to 'Long' updates its value."""
    app.tabs[1].radio[0].set_value("Long")
    _rerun_with_mocks(app)

    assert app.tabs[1].radio[0].value == "Long"


def test_summarize_change_output_language(app: AppTest) -> None:
    """Changing the output language selectbox to 'French' updates its value."""
    app.tabs[1].selectbox[0].set_value("French")
    _rerun_with_mocks(app)

    assert app.tabs[1].selectbox[0].value == "French"


# -- Model load failure -------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
    """When model loading raises an exception, st.error is shown."""
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
    """When model loading fails, the Translate button is disabled."""
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

    assert at.tabs[0].button[0].disabled


def test_model_load_failure_disables_summarize_button() -> None:
    """When model loading fails, the Summarize button is disabled."""
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

    assert at.tabs[1].button[0].disabled


# -- Result divider -----------------------------------------------------------


def test_translate_success_shows_result_divider() -> None:
    """After a successful translation, a divider appears before the result."""
    at = _run_inference_test(tab_index=0, input_text="Hello", decode_result="Bonjour")
    # Initial load has 1 divider (between steps 1 and 2); after result there are 2
    assert len(at.tabs[0].divider) >= 2


def test_summarize_success_shows_result_divider() -> None:
    """After a successful summarization, a divider appears before the result."""
    at = _run_inference_test(
        tab_index=1, input_text="Some long text.", decode_result="A brief summary."
    )
    assert len(at.tabs[1].divider) >= 2
