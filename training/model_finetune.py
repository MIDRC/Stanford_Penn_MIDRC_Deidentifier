import os
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

def normalize_label(label):
    """Normalize label names for evaluation.
       Treats 'DATES' and 'DATE' as equivalent.
    """
    if label in ["DATES", "DATE"]:
        return "DATE"
    if label in ["UNIQUE"]:
        return "ID"
    return label

def aggregate_predictions(tokens, predictions, offset_mapping, id2label):
    """
    Aggregate contiguous tokens with the same non-"O" label into a single entity.
    """
    entities = []
    current_entity = None

    for token, pred, offsets in zip(tokens, predictions, offset_mapping):
        label = id2label[pred]
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        if current_entity is not None and label == current_entity["entity_group"]:
            current_entity["word"] += " " + token.replace("##", "")
            current_entity["end"] = offsets[1]
        else:
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                "word": token.replace("##", ""),
                "entity_group": label,
                "start": offsets[0],
                "end": offsets[1]
            }
    if current_entity:
        entities.append(current_entity)
    return entities

def compute_entity_metrics(ground_truth_entities, predicted_entities):
    """
    Computes overall and per-type precision, recall, and F1 for entity-level predictions.
    """
    gt_set = {
        (entity["start_index"], entity["end_index"], normalize_label(entity["label"]))
        for entity in ground_truth_entities
    }
    pred_set = {
        (entity["start"], entity["end"], normalize_label(entity["entity_group"]))
        for entity in predicted_entities
    }
    
    overall_tp = len(gt_set.intersection(pred_set))
    overall_fp = len(pred_set - gt_set)
    overall_fn = len(gt_set - pred_set)
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    types = set([label for _, _, label in gt_set] + [label for _, _, label in pred_set])
    per_type_metrics = {}
    for phi in types:
        gt_phi = {entity for entity in gt_set if entity[2] == phi}
        pred_phi = {entity for entity in pred_set if entity[2] == phi}
        tp = len(gt_phi.intersection(pred_phi))
        fp = len(pred_phi - gt_phi)
        fn = len(gt_phi - pred_phi)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_type_metrics[phi] = {"precision": precision, "recall": recall, "f1": f1}
    
    return (overall_precision, overall_recall, overall_f1), per_type_metrics

def encode_labels(offset_mapping, phi_list, label2id):
    """
    Given token offset mapping and the list of PHI spans from the JSON,
    assign a label to each token. If a token’s span overlaps a PHI span,
    assign the corresponding (normalized) label; otherwise assign "O".
    """
    labels = []
    for offsets in offset_mapping:
        start, end = offsets
        if start == end:
            labels.append(label2id["O"])
        else:
            token_label = "O"
            for phi in phi_list:
                start_phi = phi["start_index"]
                end_phi = phi["end_index"]
                if end > start_phi and start < end_phi:
                    token_label = normalize_label(phi["label"])
                    break
            labels.append(label2id.get(token_label, label2id["O"]))
    return labels

class PHIDataset(Dataset):
    """
    Custom dataset for token classification.
    Loads JSON files from a folder, tokenizes the report text, and creates token-level labels.
    """
    def __init__(self, folder, tokenizer, label2id, limit=None):
        self.examples = []
        files = [f for f in os.listdir(folder) if f.endswith(".json")]
        files = sorted(files)
        if limit is not None:
            files = random.sample(files, limit)
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            report_text = data.get("report", "")
            phi_list = data.get("phi_list", [])
            tokenized = tokenizer(report_text, truncation=True, return_offsets_mapping=True)
            labels = encode_labels(tokenized["offset_mapping"], phi_list, label2id)
            tokenized["labels"] = labels
            tokenized.pop("offset_mapping", None)
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def evaluate_model(model, tokenizer, folder_path, id2label):
    """
    Evaluates the model on all JSON files in folder_path.
    Runs inference on each report and computes entity-level metrics.
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    all_ground_truth_entities = []
    all_predicted_entities = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report_text = data.get("report", "")
        gt_phi_list = data.get("phi_list", [])

        print(f"\nProcessing file: {file_name}")
        print(f"Report snippet: {report_text[:100]}...")

        if gt_phi_list:
            print("\nGround Truth PHI:")
            for phi in gt_phi_list:
                print(f"  Text: {phi.get('text')}, Label: {phi.get('label')}, "
                      f"Start: {phi.get('start_index')}, End: {phi.get('end_index')}")
        else:
            print("\nNo ground truth PHI found.")

        # Tokenize the report text.
        encoded = tokenizer(report_text, return_tensors="pt", return_offsets_mapping=True)
        # Move inputs to the same device as the model.
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        offset_mapping = encoded["offset_mapping"][0].tolist()
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_entities = aggregate_predictions(tokens, predictions, offset_mapping, id2label)

        print("\nPredicted PHI:")
        if predicted_entities:
            for entity in predicted_entities:
                print(f"  Text: {entity['word']}, Predicted Label: {entity['entity_group']}, "
                      f"Start: {entity['start']}, End: {entity['end']}")
        else:
            print("  No PHI detected.")

        all_ground_truth_entities.extend(gt_phi_list)
        all_predicted_entities.extend(predicted_entities)
    
    if all_ground_truth_entities:
        (overall_precision, overall_recall, overall_f1), per_type_metrics = compute_entity_metrics(
            all_ground_truth_entities, all_predicted_entities
        )
        print("\nOverall PHI detection metrics:")
        print(f"Precision: {overall_precision:.4f}")
        print(f"Recall:    {overall_recall:.4f}")
        print(f"F1 Score:  {overall_f1:.4f}")
        
        print("\nPer-PHI Type Metrics:")
        for phi_type, metrics in per_type_metrics.items():
            print(f"  {phi_type}: Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    else:
        print("No ground truth PHI available for evaluation.")

def main():
    train_folder = "train_chunked_reports_unlabeled"
    eval_folder = "chunked_reports_unlabeled"
    model_name = "StanfordAIMI/stanford-deidentifier-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    
    train_dataset = PHIDataset(train_folder, tokenizer, label2id, limit=None)
    eval_dataset = PHIDataset(eval_folder, tokenizer, label2id)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        evaluation_strategy="no",  # Evaluation run separately
        logging_steps=10,
        save_steps=5000,
        seed=42,
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting fine-tuning on 100 random training samples...")
    trainer.train()
    print("Fine-tuning complete.")
    
    print("\nEvaluating fine-tuned model on evaluation dataset...")
    model.eval()
    evaluate_model(model, tokenizer, eval_folder, id2label)

if __name__ == "__main__":
    main()

