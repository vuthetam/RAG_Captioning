from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.shared.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Internal: HDF5 feature reader 
# ---------------------------------------------------------------------------

class _H5FeatureStore:
    """Đọc CLIP features từ file HDF5 theo imgid."""

    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)
        with h5py.File(self.features_path, "r") as f:
            stored_imgids = np.asarray(f["imgids"], dtype=np.int64)
            self.feature_shape = tuple(f["features"].shape[1:])
        self._imgid_to_index = {
            int(imgid): idx for idx, imgid in enumerate(stored_imgids.tolist())
        }
        self._h5_file: h5py.File | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _feature_for_imgid(self, imgid: int) -> torch.Tensor:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        idx = self._imgid_to_index[int(imgid)]
        feature = np.asarray(self._h5_file["features"][idx])
        return torch.from_numpy(feature)


# ---------------------------------------------------------------------------
# Helper: encode K mảng words thành Tensor [K, max_len]
# ---------------------------------------------------------------------------

def _encode_ctx_field(field_list, vocab: Vocabulary, max_len: int, top_k: int) -> torch.Tensor:
    """Encode K mảng words (tokens / objects / relations) thành Tensor [K, max_len].
    """
    rows = []
    for words in field_list[:top_k]:
        if isinstance(words, np.ndarray):
            words = words.tolist()
        # Tận dụng hàm có sẵn của Vocabulary (bỏ qua attention_mask vì Encoder có thể tự suy ra từ pad_idx)
        input_ids, _ = vocab.encode_from_tokens(words, max_len)
        rows.append(torch.tensor(input_ids, dtype=torch.long))
        
    while len(rows) < top_k:
        pad_ids = [vocab.pad_idx()] * max_len
        rows.append(torch.tensor(pad_ids, dtype=torch.long))
        
    return torch.stack(rows, dim=0)  # [K, max_len]


# ---------------------------------------------------------------------------
# Dataset dùng cho Train / Val
# ---------------------------------------------------------------------------

class RagV2CaptionDataset(_H5FeatureStore, Dataset):
    """Dataset V2 cho Train/Val: trả về visual features, caption tokens và RAG contexts.

    Trả về:
        visual_feature : Tensor[D]
        input_ids      : Tensor[max_length]     — GT caption (teacher forcing + loss)
        attention_mask : Tensor[max_length]
        k_ctx_tokens   : Tensor[K, max_ctx_len]
        k_ctx_objects  : Tensor[K, max_obj_len]
        k_ctx_relations: Tensor[K, max_rel_len]
    """

    def __init__(
        self,
        df,
        vocab: Vocabulary,
        features_path: str | Path,
        rag_contexts_path: str | Path,
        max_length: int = 40,
        top_k: int = 3,
        max_ctx_len: int = 40,
        max_obj_len: int = 10,
        max_rel_len: int = 10,
    ) -> None:
        _H5FeatureStore.__init__(self, features_path)
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.top_k = top_k
        self.max_ctx_len = max_ctx_len
        self.max_obj_len = max_obj_len
        self.max_rel_len = max_rel_len

        rag_df = pd.read_parquet(rag_contexts_path)
        self._rag_lookup = dict(zip(rag_df["imgid"].tolist(), rag_df.to_dict("records")))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])

        # Visual features
        visual_feature = self._feature_for_imgid(imgid)

        # Caption tokens (GT)
        input_ids, attention_mask = self.vocab.encode_from_tokens(
            row["tokens"], self.max_length
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # RAG contexts
        ctx = self._rag_lookup.get(imgid, {})
        k_ctx_tokens    = _encode_ctx_field(ctx.get("tokens",    []), self.vocab, self.max_ctx_len, self.top_k)
        k_ctx_objects   = _encode_ctx_field(ctx.get("objects",   []), self.vocab, self.max_obj_len, self.top_k)
        k_ctx_relations = _encode_ctx_field(ctx.get("relations", []), self.vocab, self.max_rel_len, self.top_k)

        return visual_feature, input_ids, attention_mask, k_ctx_tokens, k_ctx_objects, k_ctx_relations


# ---------------------------------------------------------------------------
# Dataset dùng cho Inference / Test (sinh caption)
# ---------------------------------------------------------------------------

class RagV2InferenceDataset(_H5FeatureStore, Dataset):
    """Dataset V2 cho Inference/Test: trả về visual features và RAG contexts (không có GT caption).

    Trả về:
        visual_feature : Tensor[D]
        imgid          : int
        k_ctx_tokens   : Tensor[K, max_ctx_len]
        k_ctx_objects  : Tensor[K, max_obj_len]
        k_ctx_relations: Tensor[K, max_rel_len]
    """

    def __init__(
        self,
        df,
        vocab: Vocabulary,
        features_path: str | Path,
        rag_contexts_path: str | Path,
        top_k: int = 8,
        max_ctx_len: int = 40,
        max_obj_len: int = 10,
        max_rel_len: int = 10,
    ) -> None:
        _H5FeatureStore.__init__(self, features_path)
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.top_k = top_k
        self.max_ctx_len = max_ctx_len
        self.max_obj_len = max_obj_len
        self.max_rel_len = max_rel_len

        rag_df = pd.read_parquet(rag_contexts_path)
        self._rag_lookup = dict(zip(rag_df["imgid"].tolist(), rag_df.to_dict("records")))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])

        visual_feature = self._feature_for_imgid(imgid)

        ctx = self._rag_lookup.get(imgid, {})
        k_ctx_tokens    = _encode_ctx_field(ctx.get("tokens",    []), self.vocab, self.max_ctx_len, self.top_k)
        k_ctx_objects   = _encode_ctx_field(ctx.get("objects",   []), self.vocab, self.max_obj_len, self.top_k)
        k_ctx_relations = _encode_ctx_field(ctx.get("relations", []), self.vocab, self.max_rel_len, self.top_k)

        return visual_feature, imgid, k_ctx_tokens, k_ctx_objects, k_ctx_relations
