import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def normalize_label(label):
    """Map variant labels into canonical ones."""
    if label == "DATES":
        return "DATE"
    if label == "UNIQUE":
        return "ID"
    return label

def aggregate_predictions(tokens, predictions, offset_mapping, id2label, text):
    """Aggregate token-level predictions into contiguous entity spans."""
    entities = []
    current = None

    for tok, pred, (start, end) in zip(tokens, predictions, offset_mapping):
        raw = id2label[pred]
        if raw == "O":
            if current:
                entities.append(current)
                current = None
            continue

        label = normalize_label(raw)
        if current and label == current["entity_group"]:
            current["end"] = end
            current["word"] = text[current["start"]:end]
        else:
            if current:
                entities.append(current)
            current = {
                "start": start,
                "end": end,
                "entity_group": label,
                "word": text[start:end]
            }

    if current:
        entities.append(current)
    return entities

def predict_phi_spans(text, model_path="StanfordAIMI/stanford-deidentifier-base"):
#def predict_phi_spans(text, model_path="finetuned_model_smaller_train/final"):
    """Given a text string, return predicted PHI spans using the fine-tuned model."""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Tokenize text
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    special = enc["special_tokens_mask"][0].tolist()
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    # Model inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label

    # Collect predictions for non-special tokens
    toks, ps, offs = [], [], []
    for tok, p, off, sp in zip(
        tokenizer.convert_ids_to_tokens(input_ids[0]),
        preds, offsets, special
    ):
        if sp or off == [0, 0]:
            continue
        toks.append(tok)
        ps.append(p)
        offs.append(off)

    spans = aggregate_predictions(toks, ps, offs, id2label, text)
    return spans

if __name__ == "__main__":
    # Example usage
    sample_text = "Patient John Doe was admitted on 05/12/2024 to Stanford Hospital by Dr. Peterson. He is 90 years old and can be called at 4081239876. Mary Johnson, a 96-year-old patient, visited Massachusetts General Hospital on March 3, 2023, for a follow-up with Dr. Alice Brown. Her contact number is (617) 555-7821." 
    sample_text = "Mr. Robert Lee was discharged from UCSF Medical Center on 11/08/2022 after being treated by Dr. Karen Smith. He can be reached at 415-222-7645 and resides at 123 Elm Street, San Francisco, CA."
    #sample_text = "John and Joe were admitted to UCSF Medical Center and they are both 98 years old. They were examined by Paulina."
    sample_text = "Mr. Peterson is 96 years old and can be reached at 4084975187. He was admitted to Stanford Emergency Center."
    # Run both models and display their outputs
    print("=" * 80)
    print("Model 1: StanfordAIMI/stanford-deidentifier-base")
    print("=" * 80)
    spans1 = predict_phi_spans(sample_text, model_path="StanfordAIMI/stanford-deidentifier-base")
    print("Predicted PHI spans:\n")
    for s in spans1:
        print(f"{s['entity_group']:10s} | {s['word']!r} ({s['start']}-{s['end']})")
    
    print("\n" + "=" * 80)
    print("Model 2: StanfordAIMI/stanford-deidentifier-v2")
    print("=" * 80)
    spans2 = predict_phi_spans(sample_text, model_path="StanfordAIMI/stanford-deidentifier-v2")
    print("Predicted PHI spans:\n")
    for s in spans2:
        print(f"{s['entity_group']:10s} | {s['word']!r} ({s['start']}-{s['end']})")

