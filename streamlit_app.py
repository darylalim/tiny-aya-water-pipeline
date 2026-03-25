from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# -- Config ------------------------------------------------------------------

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

MODEL_ID: str = os.getenv("MODEL_ID", "CohereLabs/tiny-aya-water")
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "700"))
DEVICE: str = os.getenv("DEVICE", "cpu")
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
MAX_BATCH_ROWS: int = int(os.getenv("MAX_BATCH_ROWS", "100"))

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


def extract_translation(decoded_text: str) -> str:
    """Clean up decoded model output (skip_special_tokens=True already applied)."""
    return decoded_text.strip()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model: object,
    tokenizer: object,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Translate text using the model and return the cleaned result."""
    messages = build_translation_prompt(text, source_lang, target_lang)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
    )
    # Decode only the newly generated tokens (skip the input prompt)
    output_tokens = gen_tokens[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return extract_translation(decoded)


def parse_uploaded_file(
    file: BytesIO, column: str | None, max_rows: int = MAX_BATCH_ROWS
) -> list[str]:
    """Extract a list of text strings from an uploaded CSV or TXT file."""
    name: str = getattr(file, "name", "")
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
        except Exception:
            return []
        if column and column in df.columns:
            texts = df[column].astype(str).tolist()
        elif column:
            return []
        else:
            texts = df.iloc[:, 0].astype(str).tolist()
    else:
        raw = file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        texts = raw.splitlines()

    # Filter empty rows and truncate
    texts = [t for t in texts if t.strip()]
    return texts[:max_rows]


import streamlit as st


@st.cache_resource
def load_model() -> tuple:
    """Load tokenizer and model once, cached for the session lifetime."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=DEVICE
    )
    return tokenizer, model


# -- Sidebar ------------------------------------------------------------------

st.sidebar.title("Settings")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05
)
max_tokens = st.sidebar.slider(
    "Max New Tokens", min_value=100, max_value=2000, value=DEFAULT_MAX_TOKENS, step=10
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)  \n"
    "License: CC-BY-NC (non-commercial)"
)

# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model... this may take a few minutes on first run."):
        tokenizer, model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Water Translator")
st.markdown(
    "Translate between 43 European and Asia-Pacific languages using "
    "[CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) "
    "running locally."
)

# Language selectors
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Source Language", LANGUAGES, index=LANGUAGES.index("English"))
with col2:
    target_lang = st.selectbox("Target Language", LANGUAGES, index=LANGUAGES.index("French"))

# Single text translation
input_text = st.text_area("Text to translate", height=150)

if st.button("Translate", disabled=not model_loaded):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_lang == target_lang:
        st.warning("Source and target language are the same.")
    else:
        with st.spinner("Translating..."):
            result = translate_text(
                input_text, source_lang, target_lang, model, tokenizer, temperature, max_tokens
            )
        st.text_area("Translation", value=result, height=150, disabled=True)

# -- Batch Translation --------------------------------------------------------

st.markdown("---")
st.subheader("Batch Translation")

uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    # Column selector for CSV
    column: str | None = None
    if uploaded_file.name.endswith(".csv"):
        preview_df = pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="replace")
        uploaded_file.seek(0)  # Reset for re-read
        column = st.selectbox("Column to translate", preview_df.columns.tolist())

    if st.button("Translate File", disabled=not model_loaded):
        if source_lang == target_lang:
            st.warning("Source and target language are the same.")
        else:
            texts = parse_uploaded_file(uploaded_file, column=column)
            if not texts:
                st.warning("No text found in the uploaded file.")
            else:
                if len(texts) >= MAX_BATCH_ROWS:
                    st.warning(
                        f"File exceeds {MAX_BATCH_ROWS} rows. Only the first {MAX_BATCH_ROWS} will be translated."
                    )
                translations: list[str] = []
                progress = st.progress(0)
                for i, text in enumerate(texts):
                    translated = translate_text(
                        text, source_lang, target_lang, model, tokenizer, temperature, max_tokens
                    )
                    translations.append(translated)
                    progress.progress((i + 1) / len(texts))

                result_df = pd.DataFrame({"original": texts, "translated": translations})
                st.dataframe(result_df)

                csv_output = result_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_output,
                    file_name="translations.csv",
                    mime="text/csv",
                )
