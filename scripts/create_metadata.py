
import os
import pandas as pd
from pathlib import Path

# The user for whom to create the metadata
USER_NAME = "Furkan"

# Path to the user's audio data
AUDIO_DATA_PATH = Path(f"data/users/{USER_NAME}/words")

# Path to save the output metadata file
METADATA_OUTPUT_PATH = Path(f"data/users/{USER_NAME}/metadata.csv")

def create_metadata_for_user():
    """
    Scans the user's audio directory, extracts transcriptions from filenames,
    and creates a metadata.csv file.
    """
    print(f"Starting metadata creation for user: {USER_NAME}")
    print(f"Scanning audio files in: {AUDIO_DATA_PATH}")

    if not AUDIO_DATA_PATH.exists():
        print(f"Error: Audio data path not found at {AUDIO_DATA_PATH}")
        return

    metadata = []
    
    # Iterate over all .wav files in the directory
    for wav_file in AUDIO_DATA_PATH.glob("*.wav"):
        file_name = wav_file.name
        
        # Extract the transcription from the filename
        # Example: "Furkan_kelime_1_rep1.wav" -> "kelime_1"
        try:
            parts = file_name.split('_')
            transcription = f"{parts[1]}_{parts[2]}"
        except IndexError:
            print(f"Warning: Could not parse filename {file_name}. Skipping.")
            continue
            
        # Get the absolute path for the file
        absolute_file_path = wav_file.resolve()

        metadata.append({
            "file_path": str(absolute_file_path),
            "transcription": transcription
        })

    if not metadata:
        print("No .wav files found. Metadata file not created.")
        return

    # Create a pandas DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Save the DataFrame to a CSV file
    METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(METADATA_OUTPUT_PATH, index=False)
    
    print(f"Successfully created metadata file with {len(metadata)} entries.")
    print(f"File saved to: {METADATA_OUTPUT_PATH}")

if __name__ == "__main__":
    create_metadata_for_user()
