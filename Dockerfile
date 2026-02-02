# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app/Stanford_Penn_MIDRC_Deidentifier

# Download and install dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the conversion script and entrypoint script into the container
COPY convert_csv_to_npy.py .
COPY create_outputs.py .
COPY entrypoint.sh .

# Give execute permission to the entrypoint script
RUN chmod +x entrypoint.sh

# Run the script when the container launches
ENTRYPOINT ["./entrypoint.sh"]