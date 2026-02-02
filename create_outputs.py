import numpy as np
import pandas as pd
import argparse

def combine_outputs(input_csv, deidentified_npy, output_csv, output_review_csv):
    # Load the original IDs and reports
    df = pd.read_csv(input_csv)
    df_review = pd.read_csv('deidentification_details_for_review.csv')
    df_review['id'] = df['id']

    # Load the deidentified reports
    deidentified_reports = np.load(deidentified_npy, allow_pickle=True)

    # Combine the original IDs with the deidentified reports
    df['deidentified_report'] = deidentified_reports
    df.drop(columns=['report'], inplace=True)

    # Save the final output
    df.to_csv(output_csv, index=False)
    df_review.to_csv(output_review_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine deidentified reports with original IDs')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--deidentified_npy', type=str, required=True, help='Deidentified NPY file path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--output_review_csv', type=str, required=True, help='Output review CSV file path')
    args = parser.parse_args()

    combine_outputs(args.input_csv, args.deidentified_npy, args.output_csv, args.output_review_csv)
