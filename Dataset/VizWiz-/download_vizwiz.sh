#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Dataset URLs
URLS=(
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"
    "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
)

# Output directory
OUTPUT_DIR="vizwiz_dataset"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Loop through each URL
for URL in "${URLS[@]}"; do
    FILE_NAME=$(basename "$URL")          # e.g., train.zip
    DIR_NAME="${FILE_NAME%.zip}"          # e.g., train

    echo "👉 Downloading $FILE_NAME ..."
    wget -O "$OUTPUT_DIR/$FILE_NAME" "$URL"

    echo "✅ Extracting $FILE_NAME into $OUTPUT_DIR/$DIR_NAME ..."
    mkdir -p "$OUTPUT_DIR/$DIR_NAME"
    unzip -q "$OUTPUT_DIR/$FILE_NAME" -d "$OUTPUT_DIR/$DIR_NAME"

    echo "🗑️ Removing $FILE_NAME to save space..."
    rm "$OUTPUT_DIR/$FILE_NAME"
done

echo "🎉 Done! The dataset has been downloaded and extracted into: $OUTPUT_DIR"
