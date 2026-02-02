@echo off

REM Parse input arguments
set INPUT_FOLDER=%1
set DEVICE=%2

REM Build the Docker image
docker build -t deidentifier .

REM Run the Docker container with the specified arguments
docker run ^
  -v "%cd%":/usr/src/app/Stanford_Penn_MIDRC_Deidentifier ^
  -v "%INPUT_FOLDER%":/data ^
  deidentifier ^
  /data/input.csv ^
  /data/output.csv ^
  /data/output_review.csv ^
  "%DEVICE%"

REM Remove temporary files
if exist deidentification_details_for_review.csv del deidentification_details_for_review.csv
if exist reports.npy del reports.npy
if exist reports_deidentified.npy del reports_deidentified.npy
if exist deidentification_details_for_review_reports.npy del deidentification_details_for_review_reports.npy
if exist deidentification_details_for_review_labeled_reports.npy del deidentification_details_for_review_labeled_reports.npy
if exist deidentification_details_for_review_phi_lengths.npy del deidentification_details_for_review_phi_lengths.npy