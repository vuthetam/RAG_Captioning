import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset

from src.shared.vocabulary import Vocabulary

# ==========================================
# 1. Base Classes (Self-contained for V3)
# ==========================================
class _H5FeatureStoreV3:
    '''Lazily reads CLIP features from HDF5 format to avoid multi-processing issues.'''
    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)
        self._h5_file = None
        with h5py.File(self.features_path, "r") as h5_file:
            stored_imgids = np.asarray(h5_file["imgids"], dtype=np.int64)
            self.feature_shape = tuple(h5_file["features"].shape[1:])

        self._imgid_to_index = {
            int(imgid): index for index, imgid in enumerate(stored_imgids.tolist())
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _feature_for_imgid(self, imgid: int) -> torch.Tensor:
        imgid = int(imgid)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        feature_index = self._imgid_to_index[imgid]
        feature = np.asarray(self._h5_file["features"][feature_index])
        return torch.from_numpy(feature)


# ==========================================
# 2. V3 Prompt Encoding Logic
# ==========================================
def encode_rag_prompts_v3(
    captions: list, 
    objects: list, 
    relations: list, 
    vocab: Vocabulary, 
    max_rag_len: int, 
    top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Đóng gói K câu truy hồi thành K chuỗi riêng biệt (FiD style).
    Output: rag_ids [K, max_rag_len], rag_mask [K, max_rag_len]
    '''
    rag_ids = []
    rag_mask = []
    
    num_retrieved = min(top_k, len(captions))
    
    for i in range(num_retrieved):
        cap = str(captions[i]).strip()
        objs = objects[i]
        acts = relations[i]
        
        objs_str = " , ".join(objs)
        acts_str = " , ".join(acts)
        
        prompt = f"similar image shows : {cap} . objects : {objs_str} . actions : {acts_str} ."
        tokens = prompt.split()
        
        if len(tokens) > max_rag_len:
            tokens = tokens[:max_rag_len]
            
        token_ids = [vocab.token_to_idx(t) for t in tokens]
        mask = [1] * len(token_ids)
        
        pad_len = max_rag_len - len(token_ids)
        if pad_len > 0:
            token_ids.extend([vocab.pad_idx()] * pad_len)
            mask.extend([0] * pad_len)
            
        rag_ids.append(token_ids)
        rag_mask.append(mask)

    while len(rag_ids) < top_k:
        rag_ids.append([vocab.pad_idx()] * max_rag_len)
        rag_mask.append([0] * max_rag_len)
        
    return torch.tensor(rag_ids, dtype=torch.long), torch.tensor(rag_mask, dtype=torch.long)


# ==========================================
# 3. Main Datasets V3
# ==========================================
class RAGFeatureCaptionDatasetV3(_H5FeatureStoreV3, Dataset):
    '''Dataset dùng cho lúc Huấn Luyện (Training) - trả về Tuple giống V1/V2'''
    def __init__(self, df, vocab: Vocabulary, features_path, rag_contexts_path, max_length, max_rag_len, top_k):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        _H5FeatureStoreV3.__init__(self, features_path)
        
        self.rag_df = pd.read_parquet(rag_contexts_path)
        self._rag_lookup = {}
        for _, row in self.rag_df.iterrows():
            imgid = int(row["imgid"])
            self._rag_lookup[imgid] = (
                row["captions"].tolist() if isinstance(row["captions"], np.ndarray) else row["captions"],
                row["objects"].tolist() if isinstance(row["objects"], np.ndarray) else row["objects"],
                row["relations"].tolist() if isinstance(row["relations"], np.ndarray) else row["relations"]
            )
            
        self.max_rag_len = max_rag_len
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self._feature_for_imgid(imgid)
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        captions, objects, relations = self._rag_lookup.get(imgid, ([], [], []))
        rag_input_ids, rag_attention_mask = encode_rag_prompts_v3(
            captions, objects, relations, self.vocab, self.max_rag_len, self.top_k
        )
        
        return visual_feature, input_ids, attention_mask, rag_input_ids, rag_attention_mask


class RAGFeatureDatasetV3(_H5FeatureStoreV3, Dataset):
    '''Dataset dùng cho lúc Sinh Câu (Inference) - trả về Tuple giống V1/V2'''
    def __init__(self, df, features_path, rag_contexts_path, vocab: Vocabulary, max_rag_len, top_k):
        self.df = df.reset_index(drop=True)
        _H5FeatureStoreV3.__init__(self, features_path)
        
        self.rag_df = pd.read_parquet(rag_contexts_path)
        self._rag_lookup = {}
        for _, row in self.rag_df.iterrows():
            imgid = int(row["imgid"])
            self._rag_lookup[imgid] = (
                row["captions"].tolist() if isinstance(row["captions"], np.ndarray) else row["captions"],
                row["objects"].tolist() if isinstance(row["objects"], np.ndarray) else row["objects"],
                row["relations"].tolist() if isinstance(row["relations"], np.ndarray) else row["relations"]
            )
            
        self.vocab = vocab
        self.max_rag_len = max_rag_len
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self._feature_for_imgid(imgid)
        
        captions, objects, relations = self._rag_lookup.get(imgid, ([], [], []))
        rag_input_ids, rag_attention_mask = encode_rag_prompts_v3(
            captions, objects, relations, self.vocab, self.max_rag_len, self.top_k
        )
        
        return visual_feature, rag_input_ids, rag_attention_mask, imgid
