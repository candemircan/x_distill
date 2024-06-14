#!/bin/bash

# Set up directories
DATA_DIR="$X_DISTILL/data/coco"
IMAGES_DIR="$DATA_DIR/images"

# Create directories if they don't exist
mkdir -p "$IMAGES_DIR"

# Download and extract image datasets
cd "$IMAGES_DIR"
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip

# Clean up downloaded zip files
rm train2017.zip
rm val2017.zip
rm test2017.zip

# Download and extract annotations
cd "$DATA_DIR"
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q annotations_trainval2017.zip

# Clean up downloaded annotation zip file
rm annotations_trainval2017.zip
