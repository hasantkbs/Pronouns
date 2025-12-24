import pandas as pd
import os
from pathlib import Path

def fix_metadata_paths_absolute(user='Furkan'):
    """
    Rewrites the file_path column in metadata_words.csv to contain correct absolute paths
    based on the current repository root and user directory structure.
    """
    repo_root = Path(os.getcwd())
    metadata_path = repo_root / 'data' / 'users' / user / 'metadata_words.csv'
    
    if not metadata_path.exists():
        print(f"[Hata] {metadata_path} bulunamadı.")
        return

    df = pd.read_csv(metadata_path)

    if 'file_path' not in df.columns:
        print(f"[Hata] 'file_path' sütunu {metadata_path} içinde bulunamadı.")
        return

    # Assuming audio files are in 'data/users/<user>/words/' relative to repo_root
    # regardless of what the current 'file_path' column contains.
    # We will extract only the filename and reconstruct the full path.
    def reconstruct_path(existing_path):
        filename = Path(existing_path).name # Get just the filename
        # Construct the correct absolute path
        return str(repo_root / 'data' / 'users' / user / 'words' / filename)

    df['file_path'] = df['file_path'].apply(reconstruct_path)

    df.to_csv(metadata_path, index=False)
    print(f"[Bilgi] {metadata_path} dosyasındaki yollar güncel mutlak yollarla güncellendi.")

if __name__ == '__main__':
    fix_metadata_paths_absolute()