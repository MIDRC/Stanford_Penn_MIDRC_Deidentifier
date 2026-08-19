import os
import json
import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
MODEL_NAME = "StanfordAIMI/stanford-deidentifier-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load test data
file_path = "train_set.csv"
df = pd.read_csv(file_path)
assert "LabeledReport" in df.columns, "Missing required column 'LabeledReport' in CSV."

# Create output directory
output_folder = "train_chunked_reports"
os.makedirs(output_folder, exist_ok=True)

# Define break tokens
break_tokens = [".", ";", ":"]

# Process each report
for report_idx, labeled_report in enumerate(df["LabeledReport"]):
    labeled_report = labeled_report.strip()

    sentences = []
    sentence = ""

    # Split report into sentences based on break tokens
    for char in labeled_report:
        sentence += char
        if char in break_tokens:
            sentences.append(sentence.strip())
            sentence = ""

    if sentence.strip():
        sentences.append(sentence.strip())  # Add any remaining sentence

    chunks = []
    current_chunk = ""
    current_chunk_tokens = []

    # Process each sentence and create chunks
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)

        # If adding this sentence exceeds the limit, save the current chunk and start a new one
        if len(current_chunk_tokens) + len(sentence_tokens) > 508:
            if current_chunk:  # Save only if there's content
                chunks.append(current_chunk.strip())

            # Start a new chunk
            current_chunk = sentence
            current_chunk_tokens = sentence_tokens
        else:
            # Append sentence to the current chunk
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            current_chunk_tokens.extend(sentence_tokens)

    # Save the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Save chunks as JSON files
    for chunk_num, chunk_text in enumerate(chunks):
        file_name = os.path.join(output_folder, f"{report_idx}_{chunk_num}.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({"labeled_report": chunk_text}, f, indent=4)

# Print completion message
print(f"Chunking complete! Chunks saved in '{output_folder}'")

