import pandas as pd
import os

# Define the user and base path
USER = "Furkan"
USER_PATH = os.path.abspath(f"data/users/{USER}")
METADATA_PATH = os.path.join(USER_PATH, "metadata_words.csv")
WORDS_PATH = os.path.join(USER_PATH, "words")

# 1. Read the metadata
df = pd.read_csv(METADATA_PATH)
initial_count = len(df)
print(f"Initial record count: {initial_count}")

# 2. Get the list of actual audio files
try:
    existing_files = set(os.listdir(WORDS_PATH))
    print(f"Found {len(existing_files)} audio files in '{WORDS_PATH}'")
except FileNotFoundError:
    print(f"Error: Directory not found at '{WORDS_PATH}'")
    exit()

# 3. Filter the DataFrame
# The 'file_path' in the CSV is like 'words/filename.wav', so we extract the filename part
df['filename'] = df['file_path'].apply(os.path.basename)
original_columns = df.columns.tolist() # save original columns
df_filtered = df[df['filename'].isin(existing_files)]
final_count = len(df_filtered)

# Restore original columns
df_filtered = df_filtered[original_columns]


# 4. Report and save
print(f"Records remaining after validation: {final_count}")
print(f"Number of records removed: {initial_count - final_count}")

if initial_count != final_count:
    df_filtered.to_csv(METADATA_PATH, index=False)
    print(f"✅ Successfully cleaned and saved '{METADATA_PATH}'")
else:
    print("No invalid file paths found. Metadata is already clean.")

