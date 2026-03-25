from io import BytesIO
from unittest.mock import MagicMock

import torch

from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    extract_translation,
    parse_uploaded_file,
    translate_text,
)


def test_languages_list_has_43_entries() -> None:
    assert len(LANGUAGES) == 43


def test_languages_list_contains_english() -> None:
    assert "English" in LANGUAGES


def test_languages_list_contains_japanese() -> None:
    assert "Japanese" in LANGUAGES


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


def test_extract_translation_strips_whitespace() -> None:
    assert extract_translation("  Hello world  ") == "Hello world"


def test_extract_translation_empty_string() -> None:
    assert extract_translation("") == ""


def test_extract_translation_newlines() -> None:
    assert extract_translation("\n\nBonjour\n\n") == "Bonjour"


def test_extract_translation_preserves_inner_whitespace() -> None:
    assert extract_translation("  Hello   world  ") == "Hello   world"


def test_parse_uploaded_file_csv_default_column() -> None:
    csv_content = b"text,other\nhello,1\nworld,2\n"
    file = BytesIO(csv_content)
    file.name = "test.csv"
    result = parse_uploaded_file(file, column="text")
    assert result == ["hello", "world"]


def test_parse_uploaded_file_txt() -> None:
    txt_content = b"hello\nworld\n"
    file = BytesIO(txt_content)
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None)
    assert result == ["hello", "world"]


def test_parse_uploaded_file_skips_empty_rows() -> None:
    txt_content = b"hello\n\nworld\n\n"
    file = BytesIO(txt_content)
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None)
    assert result == ["hello", "world"]


def test_parse_uploaded_file_truncates_at_max_rows() -> None:
    lines = "\n".join(f"line{i}" for i in range(200)) + "\n"
    file = BytesIO(lines.encode("utf-8"))
    file.name = "test.txt"
    result = parse_uploaded_file(file, column=None, max_rows=100)
    assert len(result) == 100
    assert result[0] == "line0"
    assert result[99] == "line99"


def test_parse_uploaded_file_csv_missing_column() -> None:
    csv_content = b"text,other\nhello,1\n"
    file = BytesIO(csv_content)
    file.name = "test.csv"
    result = parse_uploaded_file(file, column="nonexistent")
    assert result == []


def test_translate_text_returns_string() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # apply_chat_template with return_tensors="pt" returns a tensor
    prompt_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = prompt_ids

    # model.generate returns prompt + generated tokens
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    # decode receives only the generated tokens (4, 5) and returns text
    mock_tokenizer.decode.return_value = "  Bonjour  "

    result = translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.1,
        max_tokens=700,
    )
    assert result == "Bonjour"


def test_translate_text_calls_generate_with_correct_params() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    prompt_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = prompt_ids
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Bonjour"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.3,
        max_tokens=500,
    )

    # Verify generate was called
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 500
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["do_sample"] is True


def test_translate_text_passes_prompt_to_tokenizer() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    prompt_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = prompt_ids
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_tokenizer.decode.return_value = "Hola"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="Spanish",
        model=mock_model,
        tokenizer=mock_tokenizer,
        temperature=0.1,
        max_tokens=700,
    )

    # Verify the prompt was built and passed to apply_chat_template
    call_args = mock_tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert "English" in messages[0]["content"]
    assert "Spanish" in messages[0]["content"]
    assert "Hello" in messages[0]["content"]
