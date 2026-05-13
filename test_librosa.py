import librosa
import pandas as pd
from pathlib import Path
import config
import os

def test_load():
    df = pd.read_csv("data/users/FurkanV1/train.csv")
    print(f"Total samples: {len(df)}")
    sample_path = df.iloc[0]["file_path"]
    print(f"Testing load: {sample_path}")
    if os.path.exists(sample_path):
        audio, sr = librosa.load(sample_path, sr=16000)
        print(f"Success! Audio shape: {audio.shape}")
    else:
        print("File does not exist!")

if __name__ == "__main__":
    test_load()
