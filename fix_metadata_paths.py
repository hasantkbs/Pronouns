import pandas as pd
import os

# Define the user and base path
USER = "Furkan"
METADATA_PATH = os.path.abspath(f"data/users/{USER}/metadata_words.csv")

def fix_path(file_path):
    """Converts absolute paths to relative paths based on the filename."""
    if os.path.isabs(file_path):
        # Extract just the filename
        filename = os.path.basename(file_path)
        # Reconstruct the correct relative path
        return f"words/{filename}"
    # If it's already a relative path, keep it as is
    return file_path

# Load the metadata
df = pd.read_csv(METADATA_PATH)

# Apply the fix to the file_path column
initial_paths = df['file_path'].copy()
df['file_path'] = df['file_path'].apply(fix_path)

# Check how many paths were changed
changed_paths = (initial_paths != df['file_path']).sum()
print(f"Total records: {len(df)}")
print(f"Number of paths corrected: {changed_paths}")

# Overwrite the original metadata file with the corrected paths
df.to_csv(METADATA_PATH, index=False)

print(f"✅ Successfully corrected paths in '{METADATA_PATH}'")
