"""
evaluate.py
-----------
Loads the trained BERT model and evaluates it on the CLINC150 test set.
Prints accuracy, F1 score, and a full classification report.

Usage:
    python evaluate.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_DIR = "./saved_model"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load the trained model and tokenizer
# ──────────────────────────────────────────────────────────────────────────────
print("Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()  # set to evaluation mode (disables dropout)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Load test dataset
# ──────────────────────────────────────────────────────────────────────────────
print("Loading CLINC150 test set...")
dataset = load_dataset("clinc_oos", "plus", split="test")
label_names = dataset.features["intent"].names

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Run predictions
# ──────────────────────────────────────────────────────────────────────────────
print("Running predictions...\n")
all_preds = []
all_labels = []

for i in range(0, len(dataset), 32):
    batch_texts = dataset[i : i + 32]["text"]
    batch_labels = dataset[i : i + 32]["intent"]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    all_preds.extend(preds)
    all_labels.extend(batch_labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Print evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy:            {accuracy:.4f}  ({accuracy:.1%})")
print(f"  F1 Score (weighted): {f1:.4f}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Full classification report
# ──────────────────────────────────────────────────────────────────────────────
print("\nClassification Report (per intent):\n")
report = classification_report(all_labels, all_preds, target_names=label_names)
print(report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write(report)

print("Report saved to classification_report.txt")
