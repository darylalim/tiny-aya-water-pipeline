from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from dotenv import load_dotenv

# -- Config ------------------------------------------------------------------

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

MODEL_ID: str = os.getenv("MODEL_ID", "CohereLabs/tiny-aya-water")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
DEVICE: str = os.getenv("DEVICE", "auto")
TOP_P: float = float(os.getenv("TOP_P", "0.95"))

# -- Languages ---------------------------------------------------------------
# Water variant: optimized for European + Asia-Pacific languages.

LANGUAGES: list[str] = [
    # European (31)
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
    "Norwegian",
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
    # Asia-Pacific (12)
    "Chinese",
    "Japanese",
    "Korean",
    "Tagalog",
    "Malay",
    "Indonesian",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Vietnamese",
    "Burmese",
]


# -- Pure functions -----------------------------------------------------------


def detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    """Pick optimal dtype for the given device."""
    import torch

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


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
    """Clean up decoded model output (skip_special_tokens=True already applied)."""
    return decoded_text.strip()


def _generate(
    messages: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    temperature: float,
    max_tokens: int,
) -> str:
    """Tokenize, generate, decode, and clean model output."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "keys"):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
    else:
        input_ids = inputs.to(model.device)
        attention_mask = None
    with torch.inference_mode():
        gen_tokens = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
        )
    # Decode only the newly generated tokens (skip the input prompt)
    output_tokens = gen_tokens[0][input_ids.shape[-1] :]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return clean_model_output(decoded)


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
    messages = build_translation_prompt(text, source_lang, target_lang)
    return _generate(messages, model, tokenizer, temperature, max_tokens)


import streamlit as st  # noqa: E402


@st.cache_resource
def load_model(device: str) -> tuple:
    """Load tokenizer and model once, cached for the session lifetime."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(device).eval()
    return tokenizer, model, device, dtype


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water Translate")


# -- Model loading ------------------------------------------------------------

try:
    _resolved_device = DEVICE if DEVICE != "auto" else detect_device()
    with st.spinner(f"Loading model on {_resolved_device.upper()}..."):
        tokenizer, model, _device, _dtype = load_model(_resolved_device)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    tokenizer, model = None, None
    model_loaded = False

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
if "doc_source_lang" not in st.session_state:
    st.session_state.doc_source_lang = "English"
if "doc_target_lang" not in st.session_state:
    st.session_state.doc_target_lang = "French"
if "doc_translated_bytes" not in st.session_state:
    st.session_state.doc_translated_bytes = b""
if "doc_translated_filename" not in st.session_state:
    st.session_state.doc_translated_filename = ""
if "_do_translate_doc" not in st.session_state:
    st.session_state._do_translate_doc = False


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


def clear_input() -> None:
    """Clear the input and output text."""
    st.session_state.translate_input = ""
    st.session_state.translate_output = ""


def swap_doc_languages() -> None:
    """Swap source/target languages in the Documents tab."""
    st.session_state.doc_source_lang, st.session_state.doc_target_lang = (
        st.session_state.doc_target_lang,
        st.session_state.doc_source_lang,
    )


def request_translate_doc() -> None:
    """Flag that a document translation was requested."""
    st.session_state._do_translate_doc = True


# -- Tabs ---------------------------------------------------------------------

tab_text, tab_docs = st.tabs(["Text", "Documents"])

# -- Text tab -----------------------------------------------------------------

with tab_text:
    # -- Language bar ---------------------------------------------------------

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

    # -- Warning slot (above panels) ------------------------------------------

    warning_slot = st.container()

    # -- Side-by-side text panels ---------------------------------------------

    col_input, col_output = st.columns(2)
    with col_input:
        translate_input = st.text_area(
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

    # -- Controls row ---------------------------------------------------------

    sub_translate, sub_clear, _, sub_copy, sub_download = st.columns(
        [6, 1, 25, 1, 1], vertical_alignment="center", gap="small"
    )
    with sub_translate:
        st.button(
            "Translate",
            key="Translate",
            on_click=request_translate,
            disabled=not model_loaded,
            type="primary",
        )
    with sub_clear:
        st.button(
            "",
            key="clear",
            icon=":material/close:",
            on_click=clear_input,
            disabled=not translate_input.strip(),
            type="tertiary",
            help="Clear source text",
        )
    with sub_copy:
        output_has_text = bool(st.session_state.translate_output.strip())
        if st.button(
            "",
            key="copy",
            icon=":material/content_copy:",
            type="tertiary",
            disabled=not output_has_text,
            help="Copy translation",
        ):
            # Use a platform-native CLI to write plain text directly to the
            # system clipboard, bypassing browser iframe clipboard API issues
            # that leak HTML and cause rich-text formatting in rich-text apps.
            _os = platform.system()
            if _os == "Darwin":
                _clip_cmd = ["/usr/bin/pbcopy"]
            elif _os == "Linux":
                _clip_cmd = ["xclip", "-selection", "clipboard"]
            else:
                _clip_cmd = ["clip"]
            try:
                subprocess.run(
                    _clip_cmd,
                    input=st.session_state.translate_output.encode("utf-8"),
                    check=True,
                )
                st.toast("Translation copied")
            except (FileNotFoundError, subprocess.CalledProcessError):
                st.warning("Could not copy to clipboard.")
    with sub_download:
        st.download_button(
            "",
            key="download",
            icon=":material/download:",
            data=st.session_state.translate_output,
            file_name="translation.txt",
            mime="text/plain",
            disabled=not output_has_text,
            type="tertiary",
            help="Download translation",
        )

    # -- Process translation request (below controls) -------------------------

    if st.session_state._do_translate:
        st.session_state._do_translate = False
        _current_input = st.session_state.translate_input
        if not _current_input.strip():
            warning_slot.warning("Please enter some text first.")
        elif st.session_state.source_lang == st.session_state.target_lang:
            warning_slot.warning("Please pick two different languages.")
        else:
            with warning_slot, st.spinner("Translating..."):
                result = translate_text(
                    _current_input,
                    st.session_state.source_lang,
                    st.session_state.target_lang,
                    model,
                    tokenizer,
                )
            st.session_state.translate_output = result
            st.rerun()  # Re-render to update the already-rendered output text_area

# -- Documents tab ------------------------------------------------------------

with tab_docs:
    doc_col_from, doc_col_swap, doc_col_to = st.columns(
        [10, 1, 10], vertical_alignment="center"
    )
    with doc_col_from:
        st.selectbox(
            "From",
            LANGUAGES,
            key="doc_source_lang",
            label_visibility="collapsed",
        )
    with doc_col_swap:
        st.button(
            "",
            key="doc_swap",
            icon=":material/swap_horiz:",
            on_click=swap_doc_languages,
            use_container_width=True,
            type="tertiary",
            help="Swap languages",
        )
    with doc_col_to:
        st.selectbox(
            "To",
            LANGUAGES,
            key="doc_target_lang",
            label_visibility="collapsed",
        )

    doc_warning_slot = st.container()

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["docx", "pdf", "pptx", "xlsx"],
        label_visibility="collapsed",
    )

    _file_too_large = (
        uploaded_file is not None and uploaded_file.size > 10 * 1024 * 1024
    )
    if _file_too_large:
        doc_warning_slot.warning("File too large. Maximum size is 10 MB.")

    st.button(
        "Translate",
        key="TranslateDoc",
        on_click=request_translate_doc,
        disabled=not model_loaded or uploaded_file is None or _file_too_large,
        type="primary",
    )

    if st.session_state._do_translate_doc:
        st.session_state._do_translate_doc = False
        if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
            doc_warning_slot.warning("Please pick two different languages.")
        elif uploaded_file is None:
            doc_warning_slot.warning("Please upload a file first.")
        else:
            from document import translate_document

            def _translate_fn(text: str) -> str:
                return translate_text(
                    text,
                    st.session_state.doc_source_lang,
                    st.session_state.doc_target_lang,
                    model,
                    tokenizer,
                )

            try:
                with doc_warning_slot, st.spinner("Translating document..."):
                    result_bytes = translate_document(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        translate_fn=_translate_fn,
                    )
            except Exception as e:
                doc_warning_slot.error(f"Failed to translate document: {e}")
            else:
                target = st.session_state.doc_target_lang
                st.session_state.doc_translated_bytes = result_bytes
                st.session_state.doc_translated_filename = (
                    f"{target}_{uploaded_file.name}"
                )
                st.rerun()

    _show_download = (
        st.session_state.doc_translated_bytes
        and uploaded_file is not None
        and st.session_state.doc_translated_filename
        == f"{st.session_state.doc_target_lang}_{uploaded_file.name}"
    )
    if _show_download:
        _mime_types = {
            ".docx": (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            ".pdf": "application/pdf",
            ".pptx": (
                "application/vnd.openxmlformats-officedocument"
                ".presentationml.presentation"
            ),
            ".xlsx": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        }
        _fname = st.session_state.doc_translated_filename
        _ext = Path(_fname).suffix.lower()
        st.download_button(
            f"Download {_fname}",
            key="doc_download",
            data=st.session_state.doc_translated_bytes,
            file_name=_fname,
            mime=_mime_types.get(_ext, "application/octet-stream"),
            type="primary",
            icon=":material/download:",
        )
