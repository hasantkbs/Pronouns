import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the user and base path
USER = "Furkan"
BASE_PATH = os.path.abspath("data/users")
USER_PATH = os.path.join(BASE_PATH, USER)

# Load the metadata
metadata_path = os.path.join(USER_PATH, "metadata_words.csv")
df = pd.read_csv(metadata_path)

# Create the full file path
df['file_path'] = df['file_path'].apply(lambda x: os.path.join(USER_PATH, x))

# Select relevant columns
df = df[['file_path', 'transcription']]

# Rename columns to match the expected format
df.rename(columns={'transcription': 'transcript'}, inplace=True)

# Split the data into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the new CSV files
train_csv_path = os.path.join(USER_PATH, "train.csv")
eval_csv_path = os.path.join(USER_PATH, "eval.csv")

train_df.to_csv(train_csv_path, index=False)
eval_df.to_csv(eval_csv_path, index=False)

print(f"Successfully created '{train_csv_path}' with {len(train_df)} records.")
print(f"Successfully created '{eval_csv_path}' with {len(eval_df)} records.")
