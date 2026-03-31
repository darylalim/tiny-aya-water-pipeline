from __future__ import annotations

import json
import os
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
def load_model() -> tuple:
    """Load tokenizer and model once, cached for the session lifetime."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = DEVICE if DEVICE != "auto" else detect_device()
    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(device).eval()
    return tokenizer, model, device, dtype


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water Translate")

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model, _device, _dtype = load_model()
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


def request_translate() -> None:
    """Flag that a translation was requested (processed before widgets)."""
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
    )
with col_to:
    target_lang = st.selectbox(
        "To",
        LANGUAGES,
        key="target_lang",
        label_visibility="collapsed",
    )

# -- Process translation request (callback + flag, before output widget) ------

warning_slot = st.container()

if st.session_state._do_translate:
    st.session_state._do_translate = False
    _current_input = st.session_state.translate_input
    if not _current_input.strip():
        warning_slot.warning("Please enter some text first.")
    elif st.session_state.source_lang == st.session_state.target_lang:
        warning_slot.warning("Please pick two different languages.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                _current_input,
                st.session_state.source_lang,
                st.session_state.target_lang,
                model,
                tokenizer,
            )
        st.session_state.translate_output = result

# -- Side-by-side text panels -------------------------------------------------

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
        key="translate_output",
        label_visibility="collapsed",
    )

# -- Controls row -------------------------------------------------------------

# Pre-reserved container so copy button's st.html() doesn't shift the controls row
clipboard_slot = st.container()
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
    )
with sub_copy:
    output_has_text = bool(st.session_state.translate_output.strip())
    if st.button(
        "",
        key="copy",
        icon=":material/content_copy:",
        type="tertiary",
        disabled=not output_has_text,
    ):
        js_text = json.dumps(st.session_state.translate_output)
        clipboard_slot.html(
            "<script>"
            "(async()=>{"
            "try{await navigator.clipboard.writeText("
            f"{js_text}"
            ")}catch{"
            "const t=document.createElement('textarea');"
            f"t.value={js_text};"
            "t.style.position='fixed';t.style.opacity='0';"
            "document.body.appendChild(t);t.select();"
            "document.execCommand('copy');"
            "document.body.removeChild(t)}"
            "})()"
            "</script>"
        )
with sub_download:
    st.download_button(
        "",
        key="download",
        icon=":material/download:",
        data=st.session_state.translate_output or " ",
        file_name="translation.txt",
        mime="text/plain",
        disabled=not output_has_text,
        type="tertiary",
    )
