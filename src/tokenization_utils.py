"""Utilities for language-aware text preprocessing before model tokenization.

This module centralizes the logic for selecting tokenizers tailored to the
low-resource languages we support. When prompts are constructed for Armenian
or Tigrinya datasets we first normalize the text using the respective
tokenizer before handing it off to the LLM-specific tokenizer (e.g., LLaMA or
OPT). Basque continues to use the default behaviour.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import Iterable, Optional

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_TIGRINYA_TOKENIZER = _REPO_ROOT / "datasets" / "TigXLNet"

if _LOCAL_TIGRINYA_TOKENIZER.exists():
    _TIGRINYA_TOKENIZER_ID = os.getenv("TIGRINYA_TOKENIZER_ID", str(_LOCAL_TIGRINYA_TOKENIZER))
else:
    _TIGRINYA_TOKENIZER_ID = os.getenv("TIGRINYA_TOKENIZER_ID", "abrhaleitela/TigXLNet")

_ARMENIAN_TOKENIZER_CLS = os.getenv("ARMENIAN_TOKENIZER_CLASS", "ArmTokenizer")

_ARMENIAN_TOKENIZER_DIR = _REPO_ROOT / "datasets" / "ArmTokenizer"

if _ARMENIAN_TOKENIZER_DIR.exists():
    arm_tok_path = str(_ARMENIAN_TOKENIZER_DIR)
    if arm_tok_path not in sys.path:
        sys.path.insert(0, arm_tok_path)


def _clean_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and trim."""

    return re.sub(r"\s+", " ", text).strip()


class _SentencePieceTokenizerWrapper:
    """Minimal wrapper to present SentencePiece with a HF-like API."""

    def __init__(self, model_path: Path):
        import sentencepiece as spm

        self._processor = spm.SentencePieceProcessor()
        self._processor.load(str(model_path))

    def tokenize(self, text: str):
        return self._processor.encode(text, out_type=str)

    def encode(self, text: str, add_special_tokens: bool = False):
        # SentencePiece does not manage special tokens; callers request raw ids.
        _ = add_special_tokens  # Unused but kept for API compatibility
        return self._processor.encode(text, out_type=int)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        _ = skip_special_tokens  # No special tokens baked in.
        return self._processor.decode(token_ids)

    def detokenize(self, tokens):
        return "".join(tokens)


@lru_cache(maxsize=1)
def _load_tigrinya_tokenizer():
    """Load and cache the tokenizer recommended for Tigrinya prompts."""

    local_dir = Path(_TIGRINYA_TOKENIZER_ID)
    spiece_path = local_dir / "spiece.model"
    if spiece_path.exists():
        return _SentencePieceTokenizerWrapper(spiece_path)

    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(_TIGRINYA_TOKENIZER_ID)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "Failed to load Tigrinya tokenizer. Set TIGRINYA_TOKENIZER_ID to a local "
            "checkpoint or ensure 'abrhaleitela/TigXLNet' is available."
        ) from exc


@lru_cache(maxsize=1)
def _load_armenian_tokenizer():
    """Load and cache the tokenizer recommended for Armenian prompts."""

    paths_to_try: Iterable[str] = (
        _ARMENIAN_TOKENIZER_CLS,
        "arm_tokenizer.ArmTokenizer",
        "ArmTokenizer.ArmTokenizer",
        "armtok.Tokenizer",
    )

    last_error: Optional[Exception] = None
    for path in paths_to_try:
        module_name, _, class_name = path.rpartition(".")
        if not module_name:
            module_name = "ArmTokenizer"
            class_name = path
        try:
            module = __import__(module_name, fromlist=[class_name])
            tokenizer_cls = getattr(module, class_name)
            return tokenizer_cls()
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            continue

    raise ImportError(
        "ArmTokenizer is required for Armenian text preprocessing. "
        "Install it from https://github.com/naymaraq/ArmTokenizer and ensure "
        "it is importable."
    ) from last_error


def _detokenize_with_fallback(tokenizer, tokens):
    """Convert tokens back to string using the best available method."""

    if hasattr(tokenizer, "detokenize"):
        detok = tokenizer.detokenize(tokens)
        if isinstance(detok, (list, tuple)):
            detok = "".join(detok)
        return detok

    if hasattr(tokenizer, "convert_tokens_to_string"):
        return tokenizer.convert_tokens_to_string(tokens)

    if hasattr(tokenizer, "decode"):
        # Some tokenizers expose `decode` that accepts token ids
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:  # pylint: disable=broad-except
            pass

    return " ".join(tokens)


def _encode_with_fallback(tokenizer, text: str):
    """Return tuple(tokens, token_ids) robustly for different tokenizers."""

    tokens = None
    token_ids = None

    if hasattr(tokenizer, "tokenize"):
        tokens = tokenizer.tokenize(text)
        if tokens is tokenizer and hasattr(tokenizer, "tokens"):
            tokens = tokenizer.tokens()
        elif hasattr(tokens, "tokens"):
            tokens = tokens.tokens()

    if hasattr(tokenizer, "encode"):
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)

    if tokens is None and token_ids is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return tokens, token_ids


def preprocess_text_for_language(text: str, language: Optional[str]) -> str:
    """Normalize text with a language-specific tokenizer when available."""

    if not text or not language:
        return text

    language = language.lower()

    if language == "tigrinya":
        tokenizer = _load_tigrinya_tokenizer()
        tokens, token_ids = _encode_with_fallback(tokenizer, text)
        if token_ids is not None and hasattr(tokenizer, "decode"):
            normalized = tokenizer.decode(token_ids, skip_special_tokens=True)
        elif tokens is not None:
            normalized = _detokenize_with_fallback(tokenizer, tokens)
        else:
            normalized = text
        return _clean_whitespace(normalized) or text

    if language == "armenian":
        tokenizer = _load_armenian_tokenizer()
        tokens, token_ids = _encode_with_fallback(tokenizer, text)
        normalized: str
        if tokens is not None:
            normalized = _detokenize_with_fallback(tokenizer, tokens)
        elif token_ids is not None and hasattr(tokenizer, "decode"):
            normalized = tokenizer.decode(token_ids)
        else:
            normalized = text
        return _clean_whitespace(normalized) or text

    return text


def dataset_to_language(dataset_name: Optional[str]) -> Optional[str]:
    """Map dataset identifiers to their language key."""

    if not dataset_name:
        return None

    dataset_name = dataset_name.lower()
    if dataset_name in {"armenian", "basque", "tigrinya"}:
        return dataset_name

    return dataset_name
