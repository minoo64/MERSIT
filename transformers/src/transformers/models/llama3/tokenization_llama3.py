# tokenization_llama3.py

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from pathlib import Path
from typing import List, Sequence, Union, Literal, AbstractSet, Optional, Tuple

import tiktoken

logger = logging.get_logger(__name__)


class Llama3Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    pretrained_vocab_files_map = {"vocab_file": {}}
    max_model_input_sizes = {"llama3": 2048}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="<|unk|>",
        pad_token=None,
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.model = self._load_vocab_file(vocab_file)

        self.bos_token_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_token_id = self.special_tokens["<|end_of_text|>"]

    def _load_vocab_file(self, vocab_file):
        assert Path(vocab_file).is_file(), f"Vocabulary file {vocab_file} not found."
        logger.info(f"Loading vocab file from {vocab_file}")

        mergeable_ranks = tiktoken.load_tiktoken_bpe(vocab_file)
        special_tokens = {
            "<|begin_of_text|>": 0,
            "<|end_of_text|>": 1,
            "<|unk|>": 2,
            # Add other special tokens as needed
        }
        return tiktoken.Encoding(
            name="llama3",
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def _convert_token_to_id(self, token: str) -> int:
        return self.model.encode(token, allowed_special={"<|begin_of_text|>", "<|end_of_text|>"})[0]

    def _convert_id_to_token(self, index: int) -> str:
        return self.model.decode([index])

    def convert_tokens_to_string(self, tokens: List[int]) -> str:
        return self.model.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids: List[int], already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return [1 if x in self.all_special_ids else 0 for x in token_ids]
        return [1] + ([0] * len(token_ids)) + [1]

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        vocab_file = Path(save_directory) / self.vocab_files_names["vocab_file"]
        logger.info(f"Saving vocabulary to {vocab_file}")
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Example: Write vocab in JSON format
            writer.write(str(self.model.get_vocab()))
        return (str(vocab_file),)

    def encode(
        self,
        s: str,
        bos: bool = True,
        eos: bool = True,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
    ) -> List[int]:
        return self.model.encode(s, allowed_special=allowed_special)

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(t)
