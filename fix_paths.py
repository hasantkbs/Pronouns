
import pandas as pd
import os

def fix_metadata_paths(user='Furkan'):
    """
    Prepends the correct relative path to the file_path column in metadata_words.csv
    for a given user.
    """
    repo_root = os.getcwd()
    metadata_path = os.path.join(repo_root, 'data', 'users', user, 'metadata_words.csv')

    if not os.path.exists(metadata_path):
        print(f"[Hata] {metadata_path} bulunamadı.")
        return

    df = pd.read_csv(metadata_path)

    if 'file_path' not in df.columns:
        print(f"[Hata] 'file_path' sütunu {metadata_path} içinde bulunamadı.")
        return

    path_prefix = f'data/users/{user}/'

    # Apply the fix only if the prefix is not already there
    df['file_path'] = df['file_path'].apply(
        lambda x: path_prefix + x if not str(x).startswith(path_prefix) else x
    )

    df.to_csv(metadata_path, index=False)
    print(f"[Bilgi] {metadata_path} dosyasındaki yollar güncellendi.")

if __name__ == '__main__':
    fix_metadata_paths()
