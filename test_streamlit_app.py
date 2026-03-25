from io import BytesIO

from streamlit_app import LANGUAGES, build_translation_prompt


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


from streamlit_app import extract_translation


def test_extract_translation_strips_whitespace() -> None:
    assert extract_translation("  Hello world  ") == "Hello world"


def test_extract_translation_empty_string() -> None:
    assert extract_translation("") == ""


def test_extract_translation_newlines() -> None:
    assert extract_translation("\n\nBonjour\n\n") == "Bonjour"


def test_extract_translation_preserves_inner_whitespace() -> None:
    assert extract_translation("  Hello   world  ") == "Hello   world"


from streamlit_app import parse_uploaded_file


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
