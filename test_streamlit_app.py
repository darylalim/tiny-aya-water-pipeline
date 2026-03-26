from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_summarization_prompt,
    build_translation_prompt,
    clean_model_output,
    detect_device,
    get_summary_config,
    parse_uploaded_file,
    select_dtype,
    translate_text,
)


def test_detect_device_returns_valid_device() -> None:
    result = detect_device()
    assert result in ("cuda", "mps", "cpu")


@patch("torch.cuda.is_available", return_value=True)
def test_detect_device_prefers_cuda(mock_cuda: MagicMock) -> None:
    assert detect_device() == "cuda"


@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.cuda.is_available", return_value=False)
def test_detect_device_falls_back_to_mps(
    mock_cuda: MagicMock, mock_mps: MagicMock
) -> None:
    assert detect_device() == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
def test_detect_device_falls_back_to_cpu(
    mock_cuda: MagicMock, mock_mps: MagicMock
) -> None:
    assert detect_device() == "cpu"


def test_select_dtype_cuda() -> None:
    assert select_dtype("cuda") == torch.bfloat16


def test_select_dtype_mps() -> None:
    assert select_dtype("mps") == torch.float16


def test_select_dtype_cpu() -> None:
    assert select_dtype("cpu") == torch.float32


def test_select_dtype_unknown_falls_back_to_float32() -> None:
    assert select_dtype("xpu") == torch.float32


@patch("streamlit_app.detect_device", return_value="mps")
def test_load_model_uses_explicit_device_over_auto(
    mock_detect: MagicMock,
) -> None:
    """When DEVICE is set explicitly, detect_device should not be called."""
    original = streamlit_app.DEVICE
    try:
        streamlit_app.DEVICE = "cpu"
        device = (
            streamlit_app.DEVICE if streamlit_app.DEVICE != "auto" else detect_device()
        )
        assert device == "cpu"
        mock_detect.assert_not_called()
    finally:
        streamlit_app.DEVICE = original


@patch("streamlit_app.detect_device", return_value="mps")
def test_load_model_calls_detect_when_auto(
    mock_detect: MagicMock,
) -> None:
    """When DEVICE is 'auto', detect_device should be called."""
    original = streamlit_app.DEVICE
    try:
        streamlit_app.DEVICE = "auto"
        device = (
            streamlit_app.DEVICE if streamlit_app.DEVICE != "auto" else detect_device()
        )
        assert device == "mps"
    finally:
        streamlit_app.DEVICE = original


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


def test_clean_model_output_strips_whitespace() -> None:
    assert clean_model_output("  Hello world  ") == "Hello world"


def test_clean_model_output_empty_string() -> None:
    assert clean_model_output("") == ""


def test_clean_model_output_newlines() -> None:
    assert clean_model_output("\n\nBonjour\n\n") == "Bonjour"


def test_clean_model_output_preserves_inner_whitespace() -> None:
    assert clean_model_output("  Hello   world  ") == "Hello   world"


def test_get_summary_config_short() -> None:
    result = get_summary_config("Short")
    assert "brief summary" in result
    assert "1-2 sentences" in result


def test_get_summary_config_medium() -> None:
    result = get_summary_config("Medium")
    assert "short paragraph" in result


def test_get_summary_config_long() -> None:
    result = get_summary_config("Long")
    assert "detailed summary" in result


def test_get_summary_config_invalid_raises() -> None:
    with pytest.raises(ValueError):
        get_summary_config("Extra Long")


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


def test_translate_text_moves_input_to_model_device() -> None:
    """Plain tensor path: apply_chat_template returns a tensor directly."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    prompt_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = prompt_ids
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_tokenizer.decode.return_value = "Bonjour"

    translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    input_ids = mock_model.generate.call_args[0][0]
    assert input_ids.device == torch.device("cpu")
    assert mock_model.generate.call_args[1]["attention_mask"] is None


def test_translate_text_handles_batch_encoding() -> None:
    """BatchEncoding path: apply_chat_template returns a dict-like object."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    prompt_ids = torch.tensor([[1, 2, 3]])
    attention = torch.tensor([[1, 1, 1]])
    batch_encoding = {"input_ids": prompt_ids, "attention_mask": attention}
    mock_tokenizer.apply_chat_template.return_value = batch_encoding

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Bonjour"

    result = translate_text(
        text="Hello",
        source_lang="English",
        target_lang="French",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    assert result == "Bonjour"
    input_ids = mock_model.generate.call_args[0][0]
    assert input_ids.device == torch.device("cpu")
    mask = mock_model.generate.call_args[1]["attention_mask"]
    assert mask.device == torch.device("cpu")
    assert torch.equal(mask, attention)


def test_translate_text_returns_string() -> None:
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

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
    mock_model.device = torch.device("cpu")

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
    mock_model.device = torch.device("cpu")

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


def test_build_summarization_prompt_returns_single_message() -> None:
    result = build_summarization_prompt("Some long text here.", "Short", "English")
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_build_summarization_prompt_contains_target_language() -> None:
    result = build_summarization_prompt("Some text.", "Medium", "French")
    content = result[0]["content"]
    assert "French" in content


def test_build_summarization_prompt_contains_input_text() -> None:
    result = build_summarization_prompt("The quick brown fox.", "Short", "English")
    content = result[0]["content"]
    assert "The quick brown fox." in content


def test_build_summarization_prompt_includes_summarize_instruction() -> None:
    result = build_summarization_prompt("Some text.", "Short", "English")
    content = result[0]["content"]
    assert "summar" in content.lower()


def test_build_summarization_prompt_includes_length_wording() -> None:
    result = build_summarization_prompt("Some text.", "Short", "English")
    content = result[0]["content"]
    assert "brief summary" in content
    assert "1-2 sentences" in content
