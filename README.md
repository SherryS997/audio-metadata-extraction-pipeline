# Audio Metadata Extraction Pipeline

This repository contains a robust pipeline for extracting various metadata from audio files, such as transcript-related information, Language Identification (LID), Speaker Diarization, Denoising, Dereverbing, Signal-to-Noise Ratio (SNR), C50, Mean Pitch, Pitch Standard Deviation (STD), and more. This pipeline leverages the `huggingface/dataspeech` repository for audio processing enrichments.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The `audio-metadata-extraction-pipeline` is designed to process audio data, specifically targeting the extraction of metadata useful for speech analysis and processing. It's built to be efficient and scalable, using multiprocessing for CPU operations and leveraging GPUs where available for accelerating computations.

## Features

- **Metadata Extraction:** Extracts transcript information (if available), LID, Speaker Diarization, Denoising, Dereverbing, SNR, C50, Mean Pitch, Pitch STD, etc.
- **Quality Estimation:** Uses `torchaudio-squim` to estimate SI-SDR, PESQ, and STOI for audio quality assessment.
- **Intermediate Storage:** Saves intermediate results to disk, allowing for resumption of interrupted processes or reuse of computed features.
- **Multiprocessing and GPU Acceleration:** Utilizes multiple CPU cores and GPUs for faster processing where applicable.
- **Flexible Input:** Accepts a JSON manifest file pointing to audio files for processing.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package installer)
- `ffmpeg` (for audio file handling)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/audio-metadata-extraction-pipeline.git
   cd audio-metadata-extraction-pipeline
   ```

2. **Install the required packages:**

   It's recommended to create a virtual environment first to avoid conflicts with system packages:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt 
   ```
   
## Usage

To run the pipeline, use the `main.py` script with the necessary arguments:

```bash
python main.py <base_dir> --audio_dir <path_to_audio_dir> --manifest_path <path_to_manifest_json> --jsonl_output_path <output_jsonl_file> --temp_dir <temp_directory> [other options]
```

**Example:**

```bash
python main.py /path/to/dataset --audio_dir audio_files --manifest_path metadata/metadata.json --jsonl_output_path output.jsonl --temp_dir temp --apply_squim_quality_estimation
```

**Arguments:**

- `base_dir`: Path to the base directory containing your dataset.
- `--audio_dir`: Path to the directory containing the audio files (relative to `base_dir`).
- `--manifest_path`: Path to the manifest JSON file listing the audio files and associated metadata.
- `--jsonl_output_path`: Path to save the output JSONL file with enriched metadata.
- `--temp_dir`: Directory to store intermediate results.
- `--apply_squim_quality_estimation`: Enable torchaudio-squim quality estimation (SI-SNR, STOI, PESQ).
- ... other optional arguments (see below for a full list)

**Optional Arguments:**

- `--output_dir`: Save the processed dataset to disk at this path.
- `--audio_column_name`: Column name in the manifest representing the audio file path (default: "audio").
- `--text_column_name`: Column name in the manifest representing the text/transcript (default: "text").
- `--rename_column`: Rename audio and text columns to 'audio' and 'text'. Useful for merging datasets.
- `--cpu_num_workers`: Number of CPU workers for CPU transformations.
- `--batch_size`: Batch size for GPU operations.
- `--cpu_writer_batch_size`: Batch size for writing results in CPU transformations.
- `--penn_batch_size`: Batch size for pitch estimation.
- `--num_workers_per_gpu_for_pitch`: Number of workers per GPU for pitch estimation.
- `--num_workers_per_gpu_for_snr`: Number of workers per GPU for SNR and reverberation estimation.
- `--num_workers_per_gpu_for_squim`: Number of workers per GPU for squim estimation.

## Pipeline Stages

The pipeline processes data in several stages, each saving intermediate results to the specified `temp_dir`:

1. **`metadata_initial`**: Loads the data from the provided JSON manifest and audio directory.
2. **`metadata_squim`** (optional): Computes SI-SDR, PESQ, and STOI if `--apply_squim_quality_estimation` is set.
3. **`metadata_pitch`**: Computes mean pitch and pitch standard deviation.
4. **`metadata_snr`**: Computes SNR, C50, and speech duration based on Voice Activity Detection (VAD).
5. **`metadata_rate`**: Computes speaking rate (if applicable).

Intermediate files are stored in `.arrow` format, allowing for efficient loading and saving of datasets.

## Configuration

Configuration parameters are passed through command-line arguments. Adjust these to suit your specific dataset and system capabilities. For example:

- Adjust `cpu_num_workers` and GPU-related worker counts based on your system resources.
- Modify `batch_size` for GPU operations to optimize memory usage and processing speed.

## Output

The pipeline outputs a JSONL (JSON Lines) file specified by `--jsonl_output_path`. Each line in the file is a JSON object representing one audio sample with its computed metadata.

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under [MIT License](LICENSE). Replace `LICENSE` with the actual license file or link if applicable.