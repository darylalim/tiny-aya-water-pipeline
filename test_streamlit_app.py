from unittest.mock import MagicMock, patch

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    clean_model_output,
    translate_text,
)

# -- LANGUAGES -----------------------------------------------------------------


def test_languages_list_has_67_entries() -> None:
    assert len(LANGUAGES) == 67


def test_languages_list_contains_english() -> None:
    assert "English" in LANGUAGES


def test_languages_list_contains_japanese() -> None:
    assert "Japanese" in LANGUAGES


# -- build_translation_prompt --------------------------------------------------


def test_build_translation_prompt_returns_single_message() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_build_translation_prompt_contains_languages() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "English" in content
    assert "French" in content


def test_build_translation_prompt_contains_text() -> None:
    result = build_translation_prompt("Good morning", "English", "Spanish")
    content = result[0]["content"]
    assert "Good morning" in content


def test_build_translation_prompt_instruction() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "Translate" in content
    assert "Output only the translation" in content


# -- clean_model_output --------------------------------------------------------


def test_clean_model_output_strips_whitespace() -> None:
    assert clean_model_output("  Hello world  ") == "Hello world"


def test_clean_model_output_empty_string() -> None:
    assert clean_model_output("") == ""


def test_clean_model_output_newlines() -> None:
    assert clean_model_output("\n\nBonjour\n\n") == "Bonjour"


def test_clean_model_output_preserves_inner_whitespace() -> None:
    assert clean_model_output("  Hello   world  ") == "Hello   world"


# -- translate_text ------------------------------------------------------------


@patch("mlx_lm.generate")
def test_translate_text_returns_cleaned_result(mock_generate: MagicMock) -> None:
    mock_generate.return_value = "  Bonjour  "
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    result = translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )
    assert result == "Bonjour"


@patch("mlx_lm.generate")
def test_translate_text_calls_generate_with_correct_params(
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Bonjour"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.3,
        max_tokens=500,
    )

    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["prompt"] == "formatted prompt"
    assert call_kwargs["max_tokens"] == 500
    assert callable(call_kwargs["sampler"])


@patch("mlx_lm.generate")
def test_translate_text_passes_prompt_to_tokenizer(
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Hola"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="Spanish",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert "English" in messages[0]["content"]
    assert "Spanish" in messages[0]["content"]
    assert "Hello" in messages[0]["content"]


@patch("mlx_lm.generate")
def test_translate_text_uses_default_params(mock_generate: MagicMock) -> None:
    mock_generate.return_value = "Bonjour"
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["max_tokens"] == streamlit_app.DEFAULT_MAX_TOKENS
    assert callable(call_kwargs["sampler"])
