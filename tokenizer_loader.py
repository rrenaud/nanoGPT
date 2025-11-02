"""
Utilities for working with interchangeable tokenizers.

Currently supports the default GPT-2 BPE from tiktoken and the HuggingFace
SuperBPE tokenizer. The goal is to offer a unified interface that can be used
by data preparation scripts and sampling utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore


@dataclass
class TokenizerAdapter:
    """Lightweight wrapper that exposes a common encode/decode API."""

    name: str
    encode_ordinary: Callable[[str], list[int]]
    encode_prompt: Callable[[str], list[int]]
    decode: Callable[[Iterable[int]], str]
    eos_token_id: int
    max_token_id: int
    vocab_size: int


def _require_dependency(package: str, handle: Optional[object]) -> None:
    if handle is None:
        raise ImportError(
            f"{package} is required for this tokenizer option but is not installed."
        )


def build_tokenizer(
    tokenizer: str,
    *,
    tokenizer_repo: Optional[str] = None,
) -> TokenizerAdapter:
    """
    Construct a tokenizer adapter for the given tokenizer identifier.

    Args:
        tokenizer: Name of the tokenizer to use. Currently supports ``'gpt2'`` and
            ``'superbpe'``.
        tokenizer_repo: Optional HuggingFace repository to source tokenizer
            artifacts from. Used for the SuperBPE option.
    """
    if tokenizer == "gpt2":
        _require_dependency("tiktoken", tiktoken)
        enc = tiktoken.get_encoding("gpt2")

        def encode_prompt(text: str) -> list[int]:
            # Allow passing through the default special token used in prompts.
            return enc.encode(text, allowed_special={"<|endoftext|>"})

        return TokenizerAdapter(
            name="gpt2",
            encode_ordinary=enc.encode_ordinary,
            encode_prompt=encode_prompt,
            decode=enc.decode,
            eos_token_id=enc.eot_token,
            max_token_id=enc.max_token_value,
            vocab_size=enc.n_vocab,
        )

    if tokenizer == "superbpe":
        _require_dependency("transformers", AutoTokenizer)
        repo_id = tokenizer_repo or "allenai/superbpe-experimental_v0.1.0"
        hf_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        eos_token_id = hf_tokenizer.eos_token_id or hf_tokenizer.bos_token_id
        if eos_token_id is None:
            raise ValueError(
                f"Tokenizer '{repo_id}' does not define an EOS or BOS token."
            )

        vocab = hf_tokenizer.get_vocab()
        max_token_id = max(vocab.values())
        max_token_id = max(max_token_id, eos_token_id)

        def encode_no_specials(text: str) -> list[int]:
            return hf_tokenizer.encode(text, add_special_tokens=False)

        def decode_tokens(tokens: Iterable[int]) -> str:
            # transformers decoders accept any iterable of ints.
            return hf_tokenizer.decode(list(tokens))

        vocab_size = max_token_id + 1

        return TokenizerAdapter(
            name="superbpe",
            encode_ordinary=encode_no_specials,
            encode_prompt=encode_no_specials,
            decode=decode_tokens,
            eos_token_id=eos_token_id,
            max_token_id=max_token_id,
            vocab_size=vocab_size,
        )

    raise ValueError(
        f"Unknown tokenizer '{tokenizer}'. Expected one of 'gpt2' or 'superbpe'."
    )
