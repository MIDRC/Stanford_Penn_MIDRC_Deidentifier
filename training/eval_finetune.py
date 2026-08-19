#!/usr/bin/env python
# evaluate_checkpoint_115000.py

import os
import json
import torch
import numpy as np
import collections
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification

def normalize_label(label):
    if label in ["DATES", "DATE"]:
        return "DATE"
    if label in ["UNIQUE"]:
        return "ID"
    return label

def aggregate_predictions(tokens, predictions, offset_mapping, id2label, report_text):
    entities = []
    current = None
    for token, pred, (start, end) in zip(tokens, predictions, offset_mapping):
        label = id2label[pred]
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue
        if current and label == current["entity_group"]:
            current["end"] = end
            current["word"] = report_text[current["start"]:current["end"]]
        else:
            if current:
                entities.append(current)
            current = {
                "start": start,
                "end": end,
                "entity_group": label,
                "word": report_text[start:end]
            }
    if current:
        entities.append(current)
    return entities

def generate_bit_map_prediction(preds, report):
    bm = np.zeros(len(report), dtype=bool)
    for p in preds:
        bm[p["start"]:p["end"]] = True
    return bm

def propagate_predictions_to_letters_around(preds, report):
    bm0 = generate_bit_map_prediction(preds, report)
    for i, pred in enumerate(preds):
        left = preds[i-1]["end"] if i > 0 else 0
        right = preds[i+1]["start"] if i < len(preds)-1 else len(report)
        s, e = pred["start"], pred["end"]
        while s > left and report[s-1].isalnum():
            s -= 1
        while e < right and report[e].isalnum():
            e += 1
        pred["start"], pred["end"] = s, e
        pred["word"] = report[s:e]
    bm1 = generate_bit_map_prediction(preds, report)
    assert not (bm0 & ~bm1).sum(), "Propagation lost coverage!"
    return preds

def fuse_continuous_predictions(preds, report):
    bm0 = generate_bit_map_prediction(preds, report)
    i = 0
    while i < len(preds)-1:
        if preds[i]["end"] == preds[i+1]["start"]:
            start_i = i
            while i+1 < len(preds) and preds[i]["end"] == preds[i+1]["start"]:
                i += 1
            end_i = i
            scores = collections.defaultdict(float)
            for j in range(start_i, end_i+1):
                lbl = preds[j].get("entity", preds[j]["entity_group"])
                span = preds[j]["end"] - preds[j]["start"]
                score = preds[j].get("score", 1.0)
                scores[lbl] += score * span
            best = max(scores, key=scores.get)
            preds[start_i].update({
                "entity": best,
                "score": scores[best],
                "end": preds[end_i]["end"],
                "word": report[preds[start_i]["start"]:preds[start_i]["end"]]
            })
            for _ in range(end_i - start_i):
                preds.pop(start_i+1)
        else:
            i += 1
    bm1 = generate_bit_map_prediction(preds, report)
    assert not (bm0 != bm1).sum(), "Fusion altered coverage!"
    return preds

def compute_metrics_from_counters(gt_counter, pred_counter):
    keys = set(gt_counter) | set(pred_counter)
    overall_tp = sum(min(gt_counter[k], pred_counter.get(k,0)) for k in keys)
    overall_fp = sum(pred_counter.values()) - overall_tp
    overall_fn = sum(gt_counter.values())     - overall_tp

    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp)>0 else 0
    recall    = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn)>0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    types = {k[2] for k in keys}
    per_type = {}
    for phi in types:
        gt_phi_vals   = {k:v for k,v in gt_counter.items()   if k[2]==phi}
        pred_phi_vals = {k:v for k,v in pred_counter.items() if k[2]==phi}

        tp_phi = sum(min(gt_phi_vals.get(k,0), pred_phi_vals.get(k,0))
                     for k in set(gt_phi_vals)|set(pred_phi_vals))
        fp_phi = sum(pred_phi_vals.values()) - tp_phi
        fn_phi = sum(gt_phi_vals.values())   - tp_phi

        p = tp_phi/(tp_phi+fp_phi) if (tp_phi+fp_phi)>0 else 0
        r = tp_phi/(tp_phi+fn_phi) if (tp_phi+fn_phi)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        per_type[phi] = {"precision":p, "recall":r, "f1":f}

    return (precision, recall, f1), per_type

def main():
    checkpoint = "./ckpt_3_epochs"
    eval_folder = "chunked_reports_unlabeled_penn"

    # Load model & tokenizer from the specific checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = model.config.id2label

    gt_counter   = Counter()
    pred_counter = Counter()

    files = sorted(f for f in os.listdir(eval_folder) if f.endswith(".json"))
    for fn in files:
        data = json.load(open(os.path.join(eval_folder, fn), encoding="utf-8"))
        rpt  = data.get("report", "")
        gt   = data.get("phi_list", [])

        # count ground truth spans
        for e in gt:
            key = (e["start_index"], e["end_index"], normalize_label(e["label"]))
            gt_counter[key] += 1

        # tokenize & predict
        enc = tokenizer(
            rpt,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0].tolist()
        spec_mask = enc["special_tokens_mask"][0].tolist()

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # filter out specials & zero-length spans
        filtered = [
            (tok, p, off)
            for tok, p, off, sp in zip(tokens, preds, offsets, spec_mask)
            if sp == 0 and off != [0, 0]
        ]

        ents = []
        if filtered:
            toks_f, prds_f, offs_f = zip(*filtered)
            ents = aggregate_predictions(toks_f, prds_f, offs_f, id2label, rpt)
            ents = propagate_predictions_to_letters_around(ents, rpt)
            ents = fuse_continuous_predictions(ents, rpt)

        # count predictions
        for e in ents:
            key = (e["start"], e["end"], normalize_label(e["entity_group"]))
            pred_counter[key] += 1

    # compute and print final metrics
    (prec, rec, f1), per_type = compute_metrics_from_counters(gt_counter, pred_counter)
    print("Evaluation results for checkpoint-115000:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}\n")
    print("Per-PHI type metrics:")
    for phi, m in per_type.items():
        print(f"  {phi}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

if __name__ == "__main__":
    main()

