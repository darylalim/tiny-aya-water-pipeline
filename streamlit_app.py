from __future__ import annotations

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


def get_summary_config(length: str) -> str:
    """Return the prompt instruction for the given summary length."""
    configs = {
        "Short": "Write a brief summary in 1-2 sentences",
        "Medium": "Write a summary in a short paragraph",
        "Long": "Write a detailed summary",
    }
    if length not in configs:
        raise ValueError(f"Unknown summary length: {length!r}")
    return configs[length]


def select_summary_length(text: str) -> str:
    """Select summary length based on input text size."""
    if len(text) > 2000:
        return "Long"
    if len(text) >= 500:
        return "Medium"
    return "Short"


def build_summarization_prompt(
    text: str, summary_length: str, target_lang: str
) -> list[dict[str, str]]:
    """Build the chat messages list for a summarization request."""
    length_instruction = get_summary_config(summary_length)
    return [
        {
            "role": "user",
            "content": (
                f"{length_instruction} of the following text in {target_lang}. "
                f"Output only the summary, nothing else.\n\n{text}"
            ),
        }
    ]


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


def summarize_text(
    text: str,
    target_lang: str,
    summary_length: str,
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Summarize text using the model and return the cleaned result."""
    messages = build_summarization_prompt(text, summary_length, target_lang)
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

st.title("Tiny Aya Water")
st.markdown("Translate and summarize text — running privately on your computer.")

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model, _device, _dtype = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    tokenizer, model = None, None
    model_loaded = False

# -- Tabs ---------------------------------------------------------------------

translate_tab, summarize_tab = st.tabs(["Translate", "Summarize"])

with translate_tab:
    col1, col2, col3 = st.columns([5, 1, 5])
    with col1:
        source_lang = st.selectbox("From", LANGUAGES)
    with col2:
        st.html(
            "<p style='text-align:center;font-size:1.2rem;padding-top:2rem'>&rarr;</p>"
        )
    with col3:
        target_lang = st.selectbox("To", LANGUAGES, index=LANGUAGES.index("French"))

    translate_input = st.text_area(
        "Text to translate",
        placeholder="Type or paste your text here...",
        height=150,
    )

    if st.button("Translate", disabled=not model_loaded):
        if not translate_input.strip():
            st.warning("Please enter some text first.")
        elif source_lang == target_lang:
            st.warning("Please pick two different languages.")
        else:
            with st.spinner("Translating..."):
                result = translate_text(
                    translate_input,
                    source_lang,
                    target_lang,
                    model,
                    tokenizer,
                )
            st.success(result)

with summarize_tab:
    output_lang = st.selectbox("Output Language", LANGUAGES)

    summarize_input = st.text_area(
        "Text to summarize",
        placeholder="Paste an article, email, or paragraph here...",
        height=150,
    )

    if st.button("Summarize", disabled=not model_loaded):
        if not summarize_input.strip():
            st.warning("Please enter some text first.")
        else:
            summary_length = select_summary_length(summarize_input)
            with st.spinner("Summarizing..."):
                result = summarize_text(
                    summarize_input,
                    output_lang,
                    summary_length,
                    model,
                    tokenizer,
                )
            st.success(result)
