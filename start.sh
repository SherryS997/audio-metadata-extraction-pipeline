#!/bin/bash

# Base directory for the dataset
BASE_DIR="/data/praveen/main/yt_sim"

lang_capitalized="$(tr '[:lower:]' '[:upper:]' <<< ${lang:0:1})${lang:1}"
echo "Processing $lang_capitalized"
mkdir -p "metadata/$lang_capitalized"
python main.py \
    "$BASE_DIR" \
    --audio_dir "$BASE_DIR/wavs" \
    --manifest_path "$BASE_DIR/metadata.json" \
    --temp_dir "$BASE_DIR/temp" \
    --output_dir "metadata" \
    --cpu_num_workers 16 \
    --batch_size 32 \
    --jsonl_output_path "metadata/metadata.json" \
    --num_workers_per_gpu_for_pitch 3 \
    --num_workers_per_gpu_for_snr 16 
    # --apply_squim_quality_estimation \
    # --num_workers_per_gpu_for_squim 16 

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Error occurred. Stopping the script."
    exit 1
fi