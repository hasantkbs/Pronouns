
import os
import pandas as pd
from pathlib import Path
import shutil

BASE_DATA_PATH = "data/users"
WHISPER_DATA_PATH = "data/whisper"

def prepare_whisper_data():
    """
    Prepares the data for Whisper fine-tuning.
    """
    print("Preparing data for Whisper fine-tuning...")

    # Create the whisper data directory
    whisper_path = Path(WHISPER_DATA_PATH)
    whisper_path.mkdir(parents=True, exist_ok=True)

    # Create the train and test directories
    train_path = whisper_path / "train"
    test_path = whisper_path / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Create the audio directories
    train_audio_path = train_path / "audio"
    test_audio_path = test_path / "audio"
    train_audio_path.mkdir(parents=True, exist_ok=True)
    test_audio_path.mkdir(parents=True, exist_ok=True)

    # Read the metadata from all users
    all_metadata = []
    for user_path in Path(BASE_DATA_PATH).iterdir():
        if user_path.is_dir():
            for metadata_file in user_path.glob("metadata*.csv"):
                try:
                    df = pd.read_csv(metadata_file)
                    all_metadata.append(df)
                except pd.errors.EmptyDataError:
                    pass

    if not all_metadata:
        print("No metadata found.")
        return

    # Concatenate all metadata
    df = pd.concat(all_metadata, ignore_index=True)

    # Split the data into train and test sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Copy the audio files and create the metadata.csv files
    for df, path, audio_path in [(train_df, train_path, train_audio_path), (test_df, test_path, test_audio_path)]:
        metadata = []
        for _, row in df.iterrows():
            # Copy the audio file
            file_path = Path(row["file_path"])
            new_file_path = audio_path / file_path.name
            shutil.copy(file_path, new_file_path)

            # Add the metadata
            metadata.append({
                "file_name": file_path.name,
                "transcription": row["transcription"]
            })

        # Create the metadata.csv file
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(path / "metadata.csv", index=False)

    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_whisper_data()
