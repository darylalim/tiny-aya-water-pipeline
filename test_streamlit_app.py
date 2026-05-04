from unittest.mock import MagicMock, patch

import numpy as np

import streamlit_app
from streamlit_app import (
    ASR_LANGUAGE_CODES,
    ASR_MODEL_ID,
    ASR_MODEL_SUBDIR,
    LANGUAGES,
    build_translation_prompt,
    clean_model_output,
    decode_audio,
    detect_speech,
    transcribe_audio,
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


def test_clean_model_output_strips_end_response_token() -> None:
    assert clean_model_output("Bonjour le monde<|END_RESPONSE|>") == "Bonjour le monde"


def test_clean_model_output_strips_end_response_token_with_whitespace() -> None:
    assert (
        clean_model_output("  Bonjour le monde  <|END_RESPONSE|>  ")
        == "Bonjour le monde"
    )


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
@patch("mlx_lm.sample_utils.make_sampler")
def test_translate_text_calls_generate_with_correct_params(
    mock_make_sampler: MagicMock,
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Bonjour"
    mock_make_sampler.return_value = MagicMock()
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

    mock_make_sampler.assert_called_once_with(temp=0.3, top_p=streamlit_app.TOP_P)
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["prompt"] == "formatted prompt"
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["sampler"] is mock_make_sampler.return_value


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
@patch("mlx_lm.sample_utils.make_sampler")
def test_translate_text_uses_default_params(
    mock_make_sampler: MagicMock,
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = "Bonjour"
    mock_make_sampler.return_value = MagicMock()
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

    mock_make_sampler.assert_called_once_with(
        temp=streamlit_app.DEFAULT_TEMPERATURE, top_p=streamlit_app.TOP_P
    )
    assert (
        mock_generate.call_args.kwargs["max_tokens"] == streamlit_app.DEFAULT_MAX_TOKENS
    )


# -- ASR_MODEL_ID --------------------------------------------------------------


def test_asr_model_id_is_mlx_8bit_repo() -> None:
    assert ASR_MODEL_ID == "mlx-community/cohere-transcribe-03-2026-mlx-8bit"


# -- ASR_MODEL_SUBDIR ----------------------------------------------------------


def test_asr_model_subdir_is_mlx_int8() -> None:
    assert ASR_MODEL_SUBDIR == "mlx-int8"


# -- ASR_LANGUAGE_CODES --------------------------------------------------------


def test_asr_language_codes_has_14_entries() -> None:
    assert len(ASR_LANGUAGE_CODES) == 14


def test_asr_language_codes_all_in_languages_list() -> None:
    assert set(ASR_LANGUAGE_CODES) <= set(LANGUAGES)


def test_asr_language_codes_english_maps_to_en() -> None:
    assert ASR_LANGUAGE_CODES["English"] == "en"


def test_asr_language_codes_chinese_maps_to_zh() -> None:
    assert ASR_LANGUAGE_CODES["Chinese"] == "zh"


def test_asr_language_codes_arabic_maps_to_ar() -> None:
    assert ASR_LANGUAGE_CODES["Arabic"] == "ar"


# -- decode_audio --------------------------------------------------------------


@patch("soundfile.read")
def test_decode_audio_downmixes_stereo(mock_sf_read: MagicMock) -> None:
    stereo = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # shape (2, 2)
    mock_sf_read.return_value = (stereo, 16000)

    result = decode_audio(b"<bytes>")

    assert result.ndim == 1
    np.testing.assert_array_almost_equal(result, np.array([0.5, 0.5]))


@patch("soundfile.read")
def test_decode_audio_resamples_when_sr_not_16khz(mock_sf_read: MagicMock) -> None:
    # 100 samples at 8000 Hz should become 200 samples at 16000 Hz
    mock_sf_read.return_value = (
        np.linspace(0.0, 1.0, 100, dtype=np.float32),
        8000,
    )

    result = decode_audio(b"<bytes>")

    assert len(result) == 200


@patch("soundfile.read")
def test_decode_audio_skips_resample_when_sr_is_16khz(
    mock_sf_read: MagicMock,
) -> None:
    audio = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    mock_sf_read.return_value = (audio, 16000)

    result = decode_audio(b"<bytes>")

    assert len(result) == 100
    np.testing.assert_array_equal(result, audio)


@patch("soundfile.read")
def test_decode_audio_returns_float32_mono_16k(mock_sf_read: MagicMock) -> None:
    mock_sf_read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 16000)

    result = decode_audio(b"<bytes>")

    assert result.dtype == np.float32
    assert result.ndim == 1


# -- transcribe_audio ----------------------------------------------------------


def test_transcribe_audio_passes_language_code() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = MagicMock(text="hola")

    transcribe_audio(audio, "Spanish", mock_model)

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] == "es"


def test_transcribe_audio_strips_whitespace() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = MagicMock(text="  hi  ")

    result = transcribe_audio(audio, "English", mock_model)

    assert result == "hi"


def test_transcribe_audio_calls_model_transcribe_once() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = MagicMock(text="ok")

    transcribe_audio(audio, "English", mock_model)

    assert mock_model.transcribe.call_count == 1


# -- detect_speech -------------------------------------------------------------


@patch("vad.vad_probabilities")
def test_detect_speech_no_speech_returns_none(mock_probs: MagicMock) -> None:
    mock_probs.return_value = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    audio = np.zeros(16000, dtype=np.float32)

    result = detect_speech(audio, vad_model={})

    assert result is None


@patch("vad.vad_probabilities")
def test_detect_speech_returns_range_for_continuous_speech(
    mock_probs: MagicMock,
) -> None:
    mock_probs.return_value = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    # 3 windows × 256 ms = 768 ms total audio
    audio = np.zeros(int(0.768 * 16000), dtype=np.float32)

    result = detect_speech(audio, vad_model={})

    assert result is not None
    start_sec, end_sec = result
    # First window starts at 0, padding clamps start to 0
    assert start_sec == 0.0
    # Last window ends at 0.768; with +200 ms pad → 0.968, but clamps to 0.768
    assert end_sec == 0.768


@patch("vad.vad_probabilities")
def test_detect_speech_returns_range_for_partial_speech(
    mock_probs: MagicMock,
) -> None:
    # Speech in middle two of four windows
    mock_probs.return_value = np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float32)
    audio = np.zeros(int(1.024 * 16000), dtype=np.float32)

    result = detect_speech(audio, vad_model={})

    assert result is not None
    start_sec, end_sec = result
    # First speech window starts at index 1 → 0.256 s; pad −200 ms → 0.056
    assert abs(start_sec - 0.056) < 1e-6
    # Last speech window ends at index 3 → 0.768 s; pad +200 ms → 0.968
    assert abs(end_sec - 0.968) < 1e-6


@patch("vad.vad_probabilities")
def test_detect_speech_applies_padding(mock_probs: MagicMock) -> None:
    # One speech window in the middle, ample audio around
    mock_probs.return_value = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32)
    audio = np.zeros(int(1.28 * 16000), dtype=np.float32)

    result = detect_speech(audio, vad_model={}, pad_seconds=0.1)

    assert result is not None
    start_sec, end_sec = result
    # Window 2 starts at 0.512 s; pad −0.1 → 0.412
    assert abs(start_sec - 0.412) < 1e-6
    # Window 2 ends at 0.768 s; pad +0.1 → 0.868
    assert abs(end_sec - 0.868) < 1e-6


@patch("vad.vad_probabilities")
def test_detect_speech_clamps_to_audio_bounds(mock_probs: MagicMock) -> None:
    # Speech in first AND last window; padding would extend past bounds
    mock_probs.return_value = np.array([0.9, 0.1, 0.1, 0.9], dtype=np.float32)
    total_dur_sec = 1.024
    audio = np.zeros(int(total_dur_sec * 16000), dtype=np.float32)

    result = detect_speech(audio, vad_model={}, pad_seconds=0.5)

    assert result is not None
    start_sec, end_sec = result
    assert start_sec == 0.0
    assert end_sec == total_dur_sec


@patch("vad.vad_probabilities")
def test_detect_speech_too_short_audio_returns_none(mock_probs: MagicMock) -> None:
    # vad_probabilities returns empty array for audio with no full block
    mock_probs.return_value = np.zeros(0, dtype=np.float32)
    audio = np.zeros(100, dtype=np.float32)  # well under 4096 samples

    result = detect_speech(audio, vad_model={})

    assert result is None
