import json
from datasets import Dataset, Audio, load_from_disk, DatasetDict
import os
import argparse, shutil
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch

def load_dataset_from_json(manifest_path, audio_dir):
    datasets = {}
    unique_filepaths = set()
    filtered_data = {
        "audio": [], 
        # "text": [], 
        "name": [], 
        "filepath": [], 
        # "duration": []
        }
    
    for split, path in [("metadata", manifest_path)]:
        path = "/".join(path.split("/")[:-1]) + f"/{split}.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        for item in data:
            filepath = f"{os.path.join(audio_dir, os.path.basename(item['filepath'])).split('.')[0]}.wav"
            if os.path.exists(filepath) and filepath not in unique_filepaths:
                filtered_data['filepath'].append(filepath)
                filtered_data['audio'].append(filepath)
                # filtered_data['text'].append(item.get('verbatim', ""))
                filtered_data['name'].append(os.path.basename(item['filepath']))
                # filtered_data['duration'].append(item.get('duration', 0.0))
        
    if len(filtered_data["audio"]) == 0:
        print(f"Skipping metadata. No files available")
        exit()

    return Dataset.from_dict(filtered_data).cast_column("audio", Audio())

def save_data(dataset, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            item_dict = dict(item)
            if 'audio' in item_dict:
                del item_dict['audio']
            for key, value in item_dict.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    item_dict[key] = str(value)
            json.dump(item_dict, f, ensure_ascii=False)
            f.write('\n')
    print(f"Dataset saved to {output_path}")

def save_intermediate(dataset, stage, temp_dir):
    dataset.save_to_disk(os.path.join(temp_dir, f"{stage}.arrow"))

def load_intermediate(stage, temp_dir):
    return load_from_disk(os.path.join(temp_dir, f"{stage}.arrow"))

def check_intermediate(stage, temp_dir):
    return os.path.exists(os.path.join(temp_dir, f"{stage}.arrow"))

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("base_dir", type=str, help="Path to the base directory containing the IVR dataset")
    # parser.add_argument("--language", type=str, required=True, help="Language to process")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files relative to base directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest JSON file")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be enriched.")
    parser.add_argument("--text_column_name", default="text", type=str, help="Text column name.")
    parser.add_argument("--rename_column", action="store_true", help="If activated, rename audio and text column names to 'audio' and 'text'. Useful if you want to merge datasets afterwards.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--batch_size", default=2, type=int, help="This parameters specify how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int, help="writer_batch_size for transformations that don't use GPUs.")
    parser.add_argument("--penn_batch_size", default=4096, type=int, help="Pitch estimation chunks audio into smaller pieces and processes them in batch. This specify the batch size.")
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int, help="Number of workers per GPU for the pitch estimation if GPUs are available.")
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int, help="Number of workers per GPU for the SNR and reverberation estimation if GPUs are available.")
    parser.add_argument("--apply_squim_quality_estimation", action="store_true", help="If set, will also use torchaudio-squim estimation (SI-SNR, STOI and PESQ).")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int, help="Number of workers per GPU for the SI-SNR, STOI and PESQ estimation if GPUs are available.")
    parser.add_argument("--jsonl_output_path", type=str, required=True, help="Path to save the output JSONL file")
    parser.add_argument("--temp_dir", type=str, required=True, help="Directory to store intermediate results.")

    args = parser.parse_args()
    args.temp_dir = f"{args.temp_dir}/metadata"

    if os.path.exists(args.jsonl_output_path):
        print(f"Skipping metadata. Already Computed")
        exit()

    os.makedirs(args.temp_dir, exist_ok=True)

    if check_intermediate(f"metadata_initial", args.temp_dir):
        print("Loading dataset from intermediate file...")
        dataset = load_intermediate(f"metadata_initial", args.temp_dir)
    else:
        dataset = load_dataset_from_json(args.manifest_path, args.audio_dir)
        save_intermediate(dataset, f"metadata_initial", args.temp_dir)
    
    audio_column_name = "audio" if args.rename_column else args.audio_column_name
    text_column_name = "text" if args.rename_column else args.text_column_name
    if args.rename_column:
        dataset = dataset.rename_columns({args.audio_column_name: "audio", args.text_column_name: "text"})

    if args.apply_squim_quality_estimation:
        if check_intermediate(f"metadata_squim", args.temp_dir):
            print("Loading squim estimation from intermediate file...")
            dataset = load_intermediate(f"metadata_squim", args.temp_dir)
        else:
            print("Compute SI-SDR, PESQ, STOI")
            dataset = dataset.map(
                squim_apply,
                batched=True,
                batch_size=args.batch_size,
                with_rank=True if torch.cuda.device_count() > 0 else False,
                num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_squim if torch.cuda.device_count() > 0 else args.cpu_num_workers,
                fn_kwargs={"audio_column_name": audio_column_name},
            )
            save_intermediate(dataset, f"metadata_squim", args.temp_dir)
    
    if check_intermediate(f"metadata_pitch", args.temp_dir):
        print("Loading pitch estimation from intermediate file...")
        dataset = load_intermediate(f"metadata_pitch", args.temp_dir)
    else:
        print("Compute pitch")
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=16_000)).map(
            pitch_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_pitch if torch.cuda.device_count() > 0 else args.cpu_num_workers,
            fn_kwargs={"audio_column_name": audio_column_name, "penn_batch_size": args.penn_batch_size},
        )
        save_intermediate(dataset, f"metadata_pitch", args.temp_dir)

    if check_intermediate(f"metadata_snr", args.temp_dir):
        print("Loading SNR and reverb estimation from intermediate file...")
        dataset = load_intermediate(f"metadata_snr", args.temp_dir)
    else:
        print("Compute snr and reverb")
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=16_000)).map(
            snr_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_snr if torch.cuda.device_count() > 0 else args.cpu_num_workers,
            fn_kwargs={"audio_column_name": audio_column_name},
        )
        save_intermediate(dataset, f"metadata_snr", args.temp_dir)

    if check_intermediate(f"metadata_rate", args.temp_dir):
        print("Loading speaking rate estimation from intermediate file...")
        dataset = load_intermediate(f"metadata_rate", args.temp_dir)
    else:
        print("Compute speaking rate")
        if "speech_duration" in dataset.features:
            dataset = dataset.map(
                rate_apply,
                with_rank=False,
                num_proc=args.cpu_num_workers,
                writer_batch_size=args.cpu_writer_batch_size,
                fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
            )
        else:
            dataset = dataset.map(
                rate_apply,
                with_rank=False,
                num_proc=args.cpu_num_workers,
                writer_batch_size=args.cpu_writer_batch_size,
                fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
            )
        save_intermediate(dataset, f"metadata_rate", args.temp_dir)

    if args.jsonl_output_path:
        save_data(dataset, args.jsonl_output_path)
        # Delete the temporary directory
        shutil.rmtree(args.temp_dir)
