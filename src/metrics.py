import json
from pathlib import Path
import pandas as pd

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def compute_metrics(predictions_path: str | Path, test_df_path: str | Path) -> dict:
    """
    Tính các chỉ số đánh giá Captioning (BLEU, METEOR, ROUGE_L, CIDEr, SPICE)
    từ file predictions.json và test_df.parquet.
    """
    print("Loading data...")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        preds_list = json.load(f)

    test_df = pd.read_parquet(test_df_path)

    res_raw = {}
    gts_raw = {}

    # Format cho PTBTokenizer: dict {imgid: [{'caption': "..."}]}
    for p in preds_list:
        imgid = int(p["imgid"])
        res_raw[imgid] = [{"caption": p["caption"]}]

    for _, row in test_df.iterrows():
        imgid = int(row["imgid"])
        if imgid in res_raw:
            gts_raw[imgid] = [{"caption": c} for c in list(row["all_raws"])]

    if not res_raw:
        raise ValueError("Không tìm thấy predictions hợp lệ!")
    
    print(f"Evaluated on {len(res_raw)} images.")

    print("Tokenizing (PTBTokenizer)...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    eval_results = {}
    
    print("Computing metrics...")
    for scorer, method in scorers:
        print(f" - Computing {method}...")
        score, _ = scorer.compute_score(gts, res)
        
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_results[m] = sc
        else:
            eval_results[method] = score

    return eval_results
