import os
import json
import re
from collections import Counter

# Directories for input and output
input_dir = "train_chunked_reports"
output_dir = "train_chunked_reports_unlabeled"

# Uncomment the following line when ready to write the files
# os.makedirs(output_dir, exist_ok=True)

# Regex pattern to match PHI in the form \PHI_TYPE[[phi content]]
phi_pattern = re.compile(r'\\(\w+)\[\[(.*?)\]\]')

# Counter to accumulate overall counts of each PHI type
phi_counts = Counter()

# Process each JSON file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Get the original report text from the JSON
        labeled_report = data.get("labeled_report", "")
        
        # Prepare to build the new report and the phi_list with positions
        phi_list = []
        new_report_parts = []
        current_pos = 0  # position in the original report
        new_report_index = 0  # current index in the new report
        
        # Iterate over each PHI marker found in the report
        for match in phi_pattern.finditer(labeled_report):
            start, end = match.span()
            
            # Append the text before the PHI marker to the new report
            preceding_text = labeled_report[current_pos:start]
            new_report_parts.append(preceding_text)
            new_report_index += len(preceding_text)
            
            # Extract PHI details: label and the inner content
            phi_label = match.group(1)
            phi_text = match.group(2)
            
            # Record this PHI occurrence with its new indices in the new report
            phi_start_index = new_report_index
            phi_end_index = new_report_index + len(phi_text)
            phi_list.append({
                "text": phi_text,
                "start_index": phi_start_index,
                "end_index": phi_end_index,
                "label": phi_label
            })
            
            # Update overall PHI counts
            phi_counts[phi_label] += 1
            
            # Append the inner phi text (without markers) to the new report
            new_report_parts.append(phi_text)
            new_report_index += len(phi_text)
            
            # Update the current position to continue after this marker
            current_pos = end
        
        # Append any remaining text after the last PHI marker
        remaining_text = labeled_report[current_pos:]
        new_report_parts.append(remaining_text)
        new_report_index += len(remaining_text)
        
        # Form the final new report string
        new_report = "".join(new_report_parts)
        
        # Prepare the new JSON structure in the desired format
        new_data = {
            "report": new_report,
            "phi_list": phi_list
        }
        
        # For sanity check, print a message for the processed file
        print(f"Processed {filename}")
        
        # Uncomment the following lines to write the new JSON file to the output directory
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(new_data, out_f, ensure_ascii=False, indent=2)

# After processing all files, print out the unique PHI types and their overall counts
print("\nUnique PHI types and overall counts:")
print("PHI Type\tCount")
for phi_type, count in phi_counts.items():
    print(f"{phi_type}\t{count}")

