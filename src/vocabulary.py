import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


class Vocabulary:
    def __init__(
        self,
        freq_threshold: int = 5,
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> None:
        self.freq_threshold = int(freq_threshold)
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.special_tokens = [pad_token, sos_token, eos_token, unk_token]
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        self._build_special_tokens()

    def _build_special_tokens(self) -> None:
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def __len__(self) -> int:
        return len(self.itos)

    def build_vocabulary(self, token_sequences: Iterable[Sequence[str]]) -> None:
        counter = Counter(token for tokens in token_sequences for token in tokens)
        for token, freq in counter.items():
            if freq >= self.freq_threshold:
                self._add_token(token)

    @classmethod
    def build_from_dataframe(
        cls,
        df,
        tokens_column: str = "tokens",
        freq_threshold: int = 5,
        **kwargs,
    ) -> "Vocabulary":
        vocab = cls(freq_threshold=freq_threshold, **kwargs)
        vocab.build_vocabulary(df[tokens_column].tolist())
        return vocab

    def token_to_idx(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx())

    def idx_to_token(self, idx: int) -> str:
        return self.itos[idx]

    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    def sos_idx(self) -> int:
        return self.stoi[self.sos_token]

    def eos_idx(self) -> int:
        return self.stoi[self.eos_token]

    def unk_idx(self) -> int:
        return self.stoi[self.unk_token]

    def encode_from_tokens(
        self,
        tokens: Sequence[str],
        max_length: int | None = None,
    ) -> tuple[list[int], list[int]]:
        token_ids = [self.sos_idx()]
        token_ids.extend(self.token_to_idx(token) for token in tokens)
        token_ids.append(self.eos_idx())

        if max_length is None:
            attention_mask = [1] * len(token_ids)
            return token_ids, attention_mask

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            token_ids[-1] = self.eos_idx()

        attention_mask = [1] * len(token_ids)
        padding_length = max_length - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.pad_idx()] * padding_length)
            attention_mask.extend([0] * padding_length)

        return token_ids, attention_mask

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> list[str]:
        tokens = [self.idx_to_token(idx) for idx in token_ids]
        if not skip_special_tokens:
            return tokens
        special_tokens = set(self.special_tokens)
        return [token for token in tokens if token not in special_tokens]

    def to_state(self) -> dict:
        return {
            "freq_threshold": self.freq_threshold,
            "pad_token": self.pad_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "special_tokens": self.special_tokens,
            "itos": self.itos,
            "stoi": self.stoi,
        }

    @classmethod
    def from_state(cls, state: dict) -> "Vocabulary":
        vocab = cls(
            freq_threshold=state.get("freq_threshold", 5),
            pad_token=state.get("pad_token", "<pad>"),
            sos_token=state.get("sos_token", "<sos>"),
            eos_token=state.get("eos_token", "<eos>"),
            unk_token=state.get("unk_token", "<unk>"),
        )
        vocab.special_tokens = list(state.get("special_tokens", vocab.special_tokens))
        vocab.itos = list(state["itos"])
        vocab.stoi = {token: int(idx) for token, idx in state["stoi"].items()}
        return vocab

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_state(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state(state)
