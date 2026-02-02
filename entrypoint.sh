#!/bin/bash

# Parse input arguments
INPUT_PATH=/data/input.csv
OUTPUT_PATH=/data/output.csv
OUTPUT_REVIEWS_PATH=/data/output_review.csv
DEVICE=$4

# Convert CSV to numpy array
python convert_csv_to_npy.py --input_path $INPUT_PATH --output_path reports.npy

# Run the deidentifier model
python main.py --input_file_path ./reports.npy --output_file_path ./reports_deidentified.npy --device $DEVICE --hospital_list stanford washington

# Combine the deidentified output with the original IDs
python create_outputs.py --input_csv $INPUT_PATH --deidentified_npy reports_deidentified.npy --output_csv $OUTPUT_PATH --output_review_csv $OUTPUT_REVIEWS_PATH