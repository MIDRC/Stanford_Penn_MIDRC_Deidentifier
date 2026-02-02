import numpy as np
import pandas as pd
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(description='Convert CSV to NPY')
parser.add_argument('--input_path', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output_path', type=str, required=True, help='Output NPY file path')
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.input_path)

# Ensure the 'report' column is in string format
reports = df['report'].astype(str).values

# Save the numpy array
np.save(args.output_path, reports)
