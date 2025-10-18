import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_clean(dataset_path: str):
    """Load and clean MedQuAD dataset."""
    df = pd.read_csv(dataset_path)
    df = df[['question', 'answer']].dropna().drop_duplicates()
    print(f"Loaded {len(df)} total examples.")
    return df

def split_dataset(df, test_size=0.15, val_size=0.5, seed=42):
    """Split data into train/val/test DataFrames."""
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    path = Path("data/medquad.csv")
    df = load_and_clean(path)
    train_df, val_df, test_df = split_dataset(df)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
