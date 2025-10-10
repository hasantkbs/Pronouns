# -*- coding: utf-8 -*-
"""
Prepare user-specific ASR training/eval CSVs from existing metadata.

- Reads data/users/<USER>/metadata_words.csv (and optionally metadata.csv)
- Normalizes file paths to the current repository root
- Renames 'transcription' -> 'transcript' to match training pipeline expectations
- Shuffles and creates a train/eval split

Usage:
    python3 scripts/prepare_user_dataset.py --user Furkan \
        --train-out data/users/Furkan/train.csv \
        --eval-out data/users/Furkan/eval.csv \
        --eval-ratio 0.1
"""

from pathlib import Path
import argparse
import pandas as pd
import random


def repo_root() -> Path:
    # scripts/ is one level under the repo root
    return Path(__file__).resolve().parent.parent


def normalize_path(p: str, root: Path) -> str:
    """Normalize a possibly foreign absolute path to the local repo layout.

    If the path contains 'Pronouns', replace the prefix up to and including
    'Pronouns' with the actual repo root path. Otherwise, if it's relative,
    resolve it against the repo root.
    """
    p_str = str(p)
    marker = "Pronouns"
    if marker in p_str:
        # Split at first occurrence of 'Pronouns' and keep the relative part after it
        prefix, _, suffix = p_str.partition(marker)
        suffix = suffix.lstrip("/")
        return str(root / suffix)
    # If already absolute and seems valid, return as-is
    if p_str.startswith("/"):
        return p_str
    # Otherwise treat as relative to repo root
    return str((root / p_str).resolve())


def load_user_frames(user: str, root: Path) -> pd.DataFrame:
    user_dir = root / "data" / "users" / user
    words_csv = user_dir / "metadata_words.csv"
    main_csv = user_dir / "metadata.csv"

    frames = []
    if words_csv.exists():
        df = pd.read_csv(words_csv)
        # Expect columns: file_path, transcription, repetition
        missing = {"file_path", "transcription"} - set(df.columns)
        if missing:
            raise ValueError(f"{words_csv} is missing required columns: {missing}")
        frames.append(df[["file_path", "transcription"]].copy())
    else:
        print(f"[Uyarı] {words_csv} bulunamadı. Kelime verisi olmadan devam ediliyor.")

    if main_csv.exists():
        df2 = pd.read_csv(main_csv)
        # Expect at least file_path, transcription
        if {"file_path", "transcription"}.issubset(df2.columns):
            frames.append(df2[["file_path", "transcription"]].copy())
        else:
            print(f"[Uyarı] {main_csv} beklenen sütunlara sahip değil, atlanıyor.")

    if not frames:
        raise FileNotFoundError(f"No suitable metadata CSVs found under {user_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Normalize paths and rename columns
    combined["file_path"] = combined["file_path"].apply(lambda x: normalize_path(x, root))
    combined = combined.rename(columns={"transcription": "transcript"})

    # Drop duplicates and rows with missing files
    combined = combined.drop_duplicates(subset=["file_path", "transcript"]).reset_index(drop=True)

    # Filter only existing files
    exists_mask = combined["file_path"].apply(lambda p: Path(p).exists())
    missing_count = (~exists_mask).sum()
    if missing_count:
        print(f"[Uyarı] {missing_count} dosya bulunamadı ve çıkarıldı.")
    combined = combined[exists_mask].reset_index(drop=True)

    if combined.empty:
        raise RuntimeError("Combined dataset is empty after path normalization and filtering.")

    return combined


def train_eval_split(df: pd.DataFrame, eval_ratio: float, seed: int = 42):
    random.seed(seed)
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_eval = max(1, int(n * eval_ratio)) if n > 1 else 0
    if n_eval >= n:
        n_eval = max(0, n - 1)
    eval_df = df.iloc[:n_eval].copy() if n_eval > 0 else pd.DataFrame(columns=df.columns)
    train_df = df.iloc[n_eval:].copy()
    return train_df, eval_df


def main():
    ap = argparse.ArgumentParser(description="Prepare user ASR train/eval CSVs")
    ap.add_argument("--user", required=True, help="User name (directory under data/users)")
    ap.add_argument("--train-out", default=None, help="Output path for train CSV")
    ap.add_argument("--eval-out", default=None, help="Output path for eval CSV")
    ap.add_argument("--eval-ratio", type=float, default=0.1, help="Portion of data reserved for eval")
    args = ap.parse_args()

    root = repo_root()

    df = load_user_frames(args.user, root)
    train_df, eval_df = train_eval_split(df, args.eval_ratio)

    # Default output locations if not provided
    user_dir = root / "data" / "users" / args.user
    train_out = Path(args.train_out) if args.train_out else (user_dir / "train.csv")
    eval_out = Path(args.eval_out) if args.eval_out else (user_dir / "eval.csv")

    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)

    if not eval_df.empty:
        eval_df.to_csv(eval_out, index=False)
        print(f"[Bilgi] train={len(train_df)}, eval={len(eval_df)} -> {train_out}, {eval_out}")
    else:
        print(f"[Bilgi] train={len(train_df)}, eval=0 -> {train_out} (eval oluşturulmadı)")


if __name__ == "__main__":
    main()