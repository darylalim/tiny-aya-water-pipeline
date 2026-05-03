from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=MagicMock(),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> None:
    """Re-run the app with mocked model loading."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=MagicMock(),
        ),
    ):
        app.run(timeout=60)


def _run_inference_test(input_text: str, generate_result: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=MagicMock(),
        ),
        patch("mlx_lm.generate", return_value=generate_result),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value(input_text)
        at.button("translate").click()
        at.run(timeout=60)
    return at


class _FakeUploadedFile:
    """Stand-in for streamlit.runtime.uploaded_file_manager.UploadedFile."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _drive_transcription(
    audio_bytes: bytes,
    transcribe_text: str,
    *,
    source_lang: str = "English",
) -> AppTest:
    """Build a fresh AppTest, simulate an upload, and return the rerun result.

    AppTest doesn't have a documented public API for driving st.file_uploader,
    so we set the widget's session_state value and the processing flag
    directly, then run the script. This exercises the transcription block
    exactly the way Streamlit's on_change callback would.
    """
    fake_model = MagicMock()
    fake_model.transcribe.return_value = MagicMock(text=transcribe_text)
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_model,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        if source_lang != "English":
            at.selectbox[0].set_value(source_lang)
            at.run(timeout=60)
        at.session_state.audio_file = _FakeUploadedFile(audio_bytes)
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "upload"
        at.run(timeout=60)
    return at


def _drive_mic_transcription(
    audio_bytes: bytes,
    transcribe_text: str,
    *,
    source_lang: str = "English",
) -> AppTest:
    """Build a fresh AppTest, simulate a mic recording, and return the rerun result.

    AppTest doesn't have a documented public API for st.audio_input, so we set
    the widget's session_state value, _do_transcribe, and _transcribe_source
    directly — same workaround pattern as _drive_transcription for files.
    """
    fake_model = MagicMock()
    fake_model.transcribe.return_value = MagicMock(text=transcribe_text)
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_model,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        if source_lang != "English":
            at.selectbox[0].set_value(source_lang)
            at.run(timeout=60)
        at.session_state.mic_input = _FakeUploadedFile(audio_bytes)
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "mic"
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
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.generate", return_value="Bonjour"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
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
    assert app.button("translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("translate").disabled


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
    assert at.text_area[1].value == "Bonjour"


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("translate").click()
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


# -- Download button -----------------------------------------------------------


def test_download_button_exists(app: AppTest) -> None:
    assert len(app.get("download_button")) == 1


def test_download_button_label(app: AppTest) -> None:
    assert app.get("download_button")[0].label == "Download"


def test_download_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.get("download_button")[0].disabled


def test_download_button_enabled_when_output_present() -> None:
    at = _run_inference_test(input_text="Hello", generate_result="Bonjour")
    assert not at.get("download_button")[0].disabled


# -- Output text area ----------------------------------------------------------


def test_output_text_area_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Model load failure --------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_model_load_failure_disables_translate_button() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("translate").disabled


# -- ASR model load -----------------------------------------------------------


def test_asr_model_load_does_not_show_error(app: AppTest) -> None:
    """When ASR loads cleanly, no error is shown."""
    error_values = [e.value for e in app.error]
    assert not any("ASR" in str(v) or "Cohere" in str(v) for v in error_values)


def test_asr_model_load_failure_shows_error() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            side_effect=RuntimeError("download failed"),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load ASR model" in str(v) for v in error_values)


# -- Audio uploader -----------------------------------------------------------


def test_audio_uploader_exists(app: AppTest) -> None:
    assert len(app.get("file_uploader")) == 1


def test_audio_uploader_enabled_for_supported_language(app: AppTest) -> None:
    # Default source is English, which is supported.
    assert not app.get("file_uploader")[0].disabled


def test_audio_uploader_disabled_for_unsupported_language(app: AppTest) -> None:
    # Hindi is in LANGUAGES but NOT in ASR_LANGUAGE_CODES.
    app.selectbox[0].set_value("Hindi")
    _rerun_with_mocks(app)

    assert app.get("file_uploader")[0].disabled


def test_audio_uploader_disabled_when_asr_load_fails() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            side_effect=RuntimeError("download failed"),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.get("file_uploader")[0].disabled


def test_audio_uploader_help_present_when_unsupported(app: AppTest) -> None:
    app.selectbox[0].set_value("Hindi")
    _rerun_with_mocks(app)

    help_text = app.get("file_uploader")[0].help or ""
    assert "Hindi" in help_text or "not supported" in help_text.lower()


def test_audio_uploader_info_visible_when_unsupported(app: AppTest) -> None:
    """A visible st.info explains why upload is unavailable for unsupported langs."""
    app.selectbox[0].set_value("Hindi")
    _rerun_with_mocks(app)

    info_values = [str(i.value) for i in app.info]
    assert any("Hindi" in v and "Audio input not supported" in v for v in info_values)


def test_audio_uploader_no_info_when_supported(app: AppTest) -> None:
    """No 'not supported' info box appears for languages Cohere Transcribe handles."""
    # Default source is English (supported).
    info_values = [str(i.value) for i in app.info]
    assert not any("not supported" in v.lower() for v in info_values)


# -- Mic widget ---------------------------------------------------------------


def test_mic_widget_exists(app: AppTest) -> None:
    assert len(app.get("audio_input")) == 1


def test_mic_widget_enabled_for_supported_language(app: AppTest) -> None:
    # Default source is English, which is supported.
    assert not app.get("audio_input")[0].disabled


def test_mic_widget_disabled_for_unsupported_language(app: AppTest) -> None:
    # Hindi is in LANGUAGES but NOT in ASR_LANGUAGE_CODES.
    app.selectbox[0].set_value("Hindi")
    _rerun_with_mocks(app)

    assert app.get("audio_input")[0].disabled


def test_mic_widget_disabled_when_asr_load_fails() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            side_effect=RuntimeError("download failed"),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.get("audio_input")[0].disabled


def test_mic_widget_help_present_when_unsupported(app: AppTest) -> None:
    app.selectbox[0].set_value("Hindi")
    _rerun_with_mocks(app)

    help_text = app.get("audio_input")[0].help or ""
    assert "Hindi" in help_text or "not supported" in help_text.lower()


def test_mic_recording_fills_input_text_area() -> None:
    at = _drive_mic_transcription(b"<bytes>", "hello world")
    assert at.text_area[0].value == "hello world"


def test_mic_transcription_failure_shows_warning() -> None:
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("kaboom")
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_model,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.mic_input = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "mic"
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Transcription failed" in str(v) for v in error_values)


def test_clear_mic_input_does_not_clear_input() -> None:
    at = _drive_mic_transcription(b"<bytes>", "hello world")
    assert at.text_area[0].value == "hello world"

    # Simulate the user clearing the recording: value becomes None,
    # but on_change still fires and sets _do_transcribe.
    at.session_state.mic_input = None
    at.session_state._do_transcribe = True
    at.session_state._transcribe_source = "mic"
    at.run(timeout=60)

    assert at.text_area[0].value == "hello world"


# -- Transcription flow -------------------------------------------------------


def test_upload_fills_input_text_area() -> None:
    at = _drive_transcription(b"<bytes>", "hello world")
    assert at.text_area[0].value == "hello world"


def test_transcription_failure_shows_warning() -> None:
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("kaboom")
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_model,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.audio_file = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "upload"
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Transcription failed" in str(v) for v in error_values)


def test_clear_uploaded_file_does_not_clear_input() -> None:
    at = _drive_transcription(b"<bytes>", "hello world")
    assert at.text_area[0].value == "hello world"

    # Simulate the user clicking the X on the uploader: file becomes None,
    # but on_change still fires and sets _do_transcribe.
    at.session_state.audio_file = None
    at.session_state._do_transcribe = True
    at.session_state._transcribe_source = "upload"
    at.run(timeout=60)

    assert at.text_area[0].value == "hello world"


def test_unsupported_language_at_upload_time_shows_warning() -> None:
    """If audio is uploaded while source language is unsupported, surface a warning.

    This is the third defense layer — the st.info banner and disabled flag
    on the uploader are upstream guards; this test exercises the runtime
    safety net inside the transcription block.
    """
    at = _drive_transcription(b"<bytes>", "irrelevant", source_lang="Hindi")

    warning_values = [str(w.value) for w in at.warning]
    assert any("not supported" in v.lower() for v in warning_values)


def test_decode_error_shows_specific_warning() -> None:
    """sf.LibsndfileError surfaces the format-specific message, not the generic one."""
    import soundfile as sf

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=MagicMock(),
        ),
        patch(
            "soundfile.read",
            side_effect=sf.LibsndfileError("Format not recognised"),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.audio_file = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "upload"
        at.run(timeout=60)

    error_values = [str(e.value) for e in at.error]
    assert any("Could not decode" in v for v in error_values)


# -- Auto-translate chain -----------------------------------------------------


def test_upload_auto_translates() -> None:
    """After a successful upload+transcription, translation runs automatically."""
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="hello")
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_asr,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
        patch("mlx_lm.generate", return_value="bonjour"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.audio_file = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "upload"
        at.run(timeout=60)

    assert at.text_area[0].value == "hello"
    assert at.text_area[1].value == "bonjour"


def test_mic_recording_auto_translates() -> None:
    """After a successful mic recording+transcription, auto-translate runs."""
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="hello")
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_asr,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
        patch("mlx_lm.generate", return_value="bonjour"),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.mic_input = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "mic"
        at.run(timeout=60)

    assert at.text_area[0].value == "hello"
    assert at.text_area[1].value == "bonjour"


def test_auto_translate_skipped_when_translation_model_not_loaded() -> None:
    """If the translation model failed to load, transcription still works but
    auto-translate is skipped (the gate prevents calling translate_text with
    model=None, which would crash)."""
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="hello")
    fake_audio = np.zeros(16000, dtype=np.float32)

    with (
        patch("mlx_lm.load", side_effect=RuntimeError("download failed")),
        patch("huggingface_hub.snapshot_download", return_value="/fake/path"),
        patch(
            "mlx_speech.generation.CohereAsrModel.from_path",
            return_value=fake_asr,
        ),
        patch("soundfile.read", return_value=(fake_audio, 16000)),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.session_state.mic_input = _FakeUploadedFile(b"<bytes>")
        at.session_state._do_transcribe = True
        at.session_state._transcribe_source = "mic"
        at.run(timeout=60)

    assert at.text_area[0].value == "hello"
    assert at.text_area[1].value == ""
