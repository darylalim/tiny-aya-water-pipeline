from __future__ import annotations

from typing import Any

import soundfile as sf

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 700
TOP_P: float = 0.95

ASR_MODEL_ID: str = "mlx-community/cohere-transcribe-03-2026-mlx-8bit"
ASR_MODEL_SUBDIR: str = "mlx-int8"  # quantization subfolder within the HF repo
ASR_LANGUAGE_CODES: dict[str, str] = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Spanish": "es",
    "Portuguese": "pt",
    "Greek": "el",
    "Dutch": "nl",
    "Polish": "pl",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Vietnamese": "vi",
    "Arabic": "ar",
}

# -- Languages ---------------------------------------------------------------
# 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa.

LANGUAGES: list[str] = [
    # Europe (31)
    "English",
    "Dutch",
    "French",
    "Italian",
    "Portuguese",
    "Romanian",
    "Spanish",
    "Czech",
    "Polish",
    "Ukrainian",
    "Russian",
    "Greek",
    "German",
    "Danish",
    "Swedish",
    "Bokmål",
    "Catalan",
    "Galician",
    "Welsh",
    "Irish",
    "Basque",
    "Croatian",
    "Latvian",
    "Lithuanian",
    "Slovak",
    "Slovenian",
    "Estonian",
    "Finnish",
    "Hungarian",
    "Serbian",
    "Bulgarian",
    # West Asia (5)
    "Arabic",
    "Persian",
    "Turkish",
    "Maltese",
    "Hebrew",
    # South Asia (9)
    "Hindi",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Punjabi",
    "Tamil",
    "Telugu",
    "Nepali",
    "Urdu",
    # Asia Pacific (12)
    "Tagalog",
    "Malay",
    "Indonesian",
    "Vietnamese",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Chinese",
    "Burmese",
    "Japanese",
    "Korean",
    # African (10)
    "Amharic",
    "Hausa",
    "Igbo",
    "Malagasy",
    "Shona",
    "Swahili",
    "Wolof",
    "Xhosa",
    "Yoruba",
    "Zulu",
]


# -- Pure functions -----------------------------------------------------------


def build_translation_prompt(
    text: str, source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    """Build the chat messages list for a translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Output only the translation, nothing else.\n\n{text}"
            ),
        }
    ]


def clean_model_output(decoded_text: str) -> str:
    """Clean up decoded model output."""
    return decoded_text.replace("<|END_RESPONSE|>", "").strip()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Translate text using the model and return the cleaned result."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = build_translation_prompt(text, source_lang, target_lang)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temperature, top_p=TOP_P)
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    return clean_model_output(result)


def transcribe_audio(
    audio_bytes: bytes,
    language: str,
    model: Any,
) -> str:
    """Decode audio bytes → mono 16 kHz float32 → transcribe → cleaned text."""
    import io

    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=False
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        old_len = len(audio)
        new_len = int(round(old_len * 16000 / sample_rate))
        audio = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            audio,
        ).astype(np.float32)
    lang_code = ASR_LANGUAGE_CODES[language]
    result = model.transcribe(audio=audio, sample_rate=16000, language=lang_code)
    return result.text.strip()


import streamlit as st  # noqa: E402


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


@st.cache_resource
def load_asr_model() -> Any:
    """Load the Cohere Transcribe MLX model once, cached for the session lifetime."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from mlx_speech.generation import CohereAsrModel

    local_dir = Path(snapshot_download(repo_id=ASR_MODEL_ID))
    return CohereAsrModel.from_path(local_dir / ASR_MODEL_SUBDIR)


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Global Translate")


# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model, tokenizer = None, None
    model_loaded = False

# -- ASR model loading --------------------------------------------------------

try:
    with st.spinner("Loading ASR model..."):
        asr_model = load_asr_model()
    asr_loaded = True
except Exception as e:
    st.error(f"Failed to load ASR model: {e}")
    asr_model = None
    asr_loaded = False

# -- Session state defaults ---------------------------------------------------

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "French"
if "translate_input" not in st.session_state:
    st.session_state.translate_input = ""
if "translate_output" not in st.session_state:
    st.session_state.translate_output = ""
if "_do_translate" not in st.session_state:
    st.session_state._do_translate = False
if "_do_transcribe" not in st.session_state:
    st.session_state._do_transcribe = False


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def request_transcribe() -> None:
    """Flag that a transcription was requested (processed after the uploader row)."""
    st.session_state._do_transcribe = True


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


# -- Language bar -------------------------------------------------------------

col_from, col_swap, col_to = st.columns([10, 1, 10], vertical_alignment="center")
with col_from:
    source_lang = st.selectbox(
        "From",
        LANGUAGES,
        key="source_lang",
        label_visibility="collapsed",
    )
with col_swap:
    st.button(
        "",
        key="swap",
        icon=":material/swap_horiz:",
        on_click=swap_languages,
        use_container_width=True,
        type="tertiary",
        help="Swap languages",
    )
with col_to:
    target_lang = st.selectbox(
        "To",
        LANGUAGES,
        key="target_lang",
        label_visibility="collapsed",
    )

# -- Audio upload row ---------------------------------------------------------

asr_supported = st.session_state.source_lang in ASR_LANGUAGE_CODES
uploader_disabled = not asr_loaded or not asr_supported
uploader_help = (
    None
    if asr_supported
    else (
        f"Audio transcription not supported for {st.session_state.source_lang}. "
        "Cohere Transcribe supports: " + ", ".join(ASR_LANGUAGE_CODES.keys())
    )
)
if not asr_loaded:
    st.info("ASR model not loaded; audio upload unavailable.")
elif not asr_supported:
    st.info(
        f"Audio upload not supported for {st.session_state.source_lang}. "
        f"Cohere Transcribe supports: {', '.join(ASR_LANGUAGE_CODES.keys())}."
    )
st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    key="audio_file",
    on_change=request_transcribe,
    disabled=uploader_disabled,
    help=uploader_help,
    label_visibility="collapsed",
)

# -- Warning slot (above panels) ---------------------------------------------

warning_slot = st.container()

# -- Process transcription request -------------------------------------------

if st.session_state._do_transcribe:
    st.session_state._do_transcribe = False
    uploaded = st.session_state.audio_file
    if uploaded is None:
        pass
    elif st.session_state.source_lang not in ASR_LANGUAGE_CODES:
        warning_slot.warning("Audio language not supported.")
    else:
        try:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(
                    uploaded.getvalue(),
                    st.session_state.source_lang,
                    asr_model,
                )
        except sf.LibsndfileError:
            warning_slot.error(
                "Could not decode audio file. Try WAV/FLAC if MP3 fails."
            )
        except Exception as e:
            warning_slot.error(f"Transcription failed: {e}")
        else:
            st.session_state.translate_input = transcript

# -- Side-by-side text panels ------------------------------------------------

col_input, col_output = st.columns(2)
with col_input:
    st.text_area(
        "Input",
        height=300,
        max_chars=5000,
        key="translate_input",
        label_visibility="collapsed",
    )
with col_output:
    st.text_area(
        "Output",
        height=300,
        placeholder="Translation",
        disabled=True,
        value=st.session_state.translate_output,
        label_visibility="collapsed",
    )

# -- Controls row -------------------------------------------------------------

sub_translate, sub_download = st.columns(2, vertical_alignment="center", gap="small")
with sub_translate:
    st.button(
        "Translate",
        key="translate",
        on_click=request_translate,
        disabled=not model_loaded,
        type="primary",
        use_container_width=True,
    )
with sub_download:
    st.download_button(
        "Download",
        key="download",
        data=st.session_state.translate_output,
        file_name="translation.txt",
        mime="text/plain",
        disabled=not st.session_state.translate_output.strip(),
        type="secondary",
        use_container_width=True,
    )

# -- Process translation request (below controls) ---------------------------

if st.session_state._do_translate:
    st.session_state._do_translate = False
    current_input = st.session_state.translate_input
    if not current_input.strip():
        warning_slot.warning("Please enter some text first.")
    elif st.session_state.source_lang == st.session_state.target_lang:
        warning_slot.warning("Please pick two different languages.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                current_input,
                st.session_state.source_lang,
                st.session_state.target_lang,
                model,
                tokenizer,
            )
        st.session_state.translate_output = result
        st.rerun()  # Re-render to update the already-rendered output text_area
