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
