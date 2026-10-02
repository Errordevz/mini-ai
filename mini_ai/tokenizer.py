"""Thin wrapper around Hugging Face Tokenizers."""
from pathlib import Path
from tokenizers import Tokenizer


class MiniTokenizer:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens=True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str | Path):
        self.tokenizer.save(str(path))
