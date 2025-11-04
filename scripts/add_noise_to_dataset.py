import os
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

def main():
    parser = argparse.ArgumentParser(description="Augment audio data by adding noise and other effects.")
    parser.add_argument("user_id", type=str, help="User ID for whom to augment the data.")
    parser.add_argument("--num_augmentations", type=int, default=2, help="Number of augmented versions to create per audio file.")
    parser.add_argument("--noise_level", type=float, default=0.005, help="Level of Gaussian noise to add.")
    args = parser.parse_args()

    user_data_path = Path("data/users") / args.user_id
    metadata_path = user_data_path / "metadata_words.csv"
    words_path = user_data_path / "words"
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=args.noise_level, p=0.8),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
    ])

    new_metadata = []

    for index, row in df.iterrows():
        filename = os.path.basename(row["file_path"])
        file_path = words_path / filename
        
        if not file_path.exists():
            print(f"Warning: Audio file not found, skipping: {file_path}")
            continue

        audio, sample_rate = sf.read(file_path)
        
        for i in range(args.num_augmentations):
            augmented_audio = augment(samples=audio, sample_rate=sample_rate)
            
            new_filename = f"{file_path.stem}_aug_{i}.wav"
            new_filepath = words_path / new_filename
            sf.write(new_filepath, augmented_audio, sample_rate)
            
            new_metadata.append({
                "file_path": new_filename,
                "transcription": row["transcription"]
            })
            print(f"Generated augmented file: {new_filepath}")

    new_df = pd.DataFrame(new_metadata)
    
    backup_path = metadata_path.with_suffix('.csv.bak')
    if not backup_path.exists():
        os.rename(metadata_path, backup_path)
        print(f"Backed up original metadata to {backup_path}")

    # Load the backup data to concat with new data
    backup_df = pd.read_csv(backup_path)
    updated_df = pd.concat([backup_df, new_df], ignore_index=True)
    updated_df.to_csv(metadata_path, index=False)
    print(f"Updated metadata file at {metadata_path} with {len(new_df)} new augmented entries.")

if __name__ == "__main__":
    main()
