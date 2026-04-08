"""
dataset_loader.py
-----------------
Loads the CLINC150 intent classification dataset.
Can load from a local CSV file (dataset.csv) or from HuggingFace.

Usage:
    from dataset_loader import load_clinc
    train_dataset, test_dataset, label_names = load_clinc()
"""

import os
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
CSV_PATH = "./dataset.csv"


def load_clinc():
    """
    Load the CLINC150 dataset and tokenize it for BERT.

    First tries to load from dataset.csv (local file).
    Falls back to downloading from HuggingFace if the file is missing.

    Returns:
        train_dataset: tokenized training set (PyTorch format)
        test_dataset:  tokenized test set (PyTorch format)
        label_names:   list of intent label strings
    """
    if os.path.exists(CSV_PATH):
        # ── Load from local CSV ──────────────────────────────────────
        print(f"Loading dataset from {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)

        # Get sorted label names for consistent encoding
        label_names = sorted(df["intent"].unique().tolist())
        label2id = {name: i for i, name in enumerate(label_names)}

        # Split into train and test
        train_df = df[df["split"] == "train"][["text", "intent"]].copy()
        test_df = df[df["split"] == "test"][["text", "intent"]].copy()

        # Encode labels as integers
        train_df["labels"] = train_df["intent"].map(label2id)
        test_df["labels"] = test_df["intent"].map(label2id)

        # Convert to HuggingFace Dataset
        train_data = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
        test_data = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False)

        print(f"Train samples: {len(train_data)}")
        print(f"Test samples:  {len(test_data)}")
        print(f"Intents:       {len(label_names)}")

    else:
        # ── Load from HuggingFace ────────────────────────────────────
        print("dataset.csv not found. Downloading from HuggingFace...")
        dataset = load_dataset("clinc_oos", "plus")

        label_names = dataset["train"].features["intent"].names
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples:  {len(dataset['test'])}")
        print(f"Intents:       {len(label_names)}")

        train_data = dataset["train"].rename_column("intent", "labels")
        test_data = dataset["test"].rename_column("intent", "labels")

    # Show a few examples
    print("\n--- Sample Data ---")
    for i in range(3):
        text = train_data[i]["text"]
        label_id = train_data[i]["labels"]
        print(f"  Text:   {text}")
        print(f"  Intent: {label_names[label_id]}")
        print()

    # Tokenize using BERT tokenizer
    print("Tokenizing with BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Dataset ready!\n")
    return train_data, test_data, label_names


if __name__ == "__main__":
    train, test, labels = load_clinc()
    print(f"Loaded {len(train)} training and {len(test)} test samples.")
    print(f"Total intents: {len(labels)}")
