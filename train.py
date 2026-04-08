"""
train.py
--------
Fine-tunes a pretrained BERT model on the CLINC150 intent classification
dataset. Uses gradient descent (AdamW optimizer) and backpropagation to
update model weights over multiple epochs.

Usage:
    python train.py
"""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from dataset_loader import load_clinc

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./saved_model"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # standard for BERT fine-tuning
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load and preprocess the dataset
# ──────────────────────────────────────────────────────────────────────────────
train_dataset, test_dataset, label_names = load_clinc()
num_labels = len(label_names)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Load pretrained BERT model for classification
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading pretrained {MODEL_NAME} with {num_labels} output classes...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Define evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    Called at the end of each epoch to compute accuracy and F1 score.
    eval_pred contains (logits, labels) from the model output.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # pick the class with highest score
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Set up training arguments
# ──────────────────────────────────────────────────────────────────────────────
# The Trainer handles the training loop internally:
#   - Forward pass: compute predictions
#   - Loss calculation: CrossEntropyLoss (built into the model)
#   - Backward pass: backpropagation to compute gradients
#   - Optimizer step: AdamW updates weights using gradient descent
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,               # L2 regularization
    eval_strategy="epoch",           # evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,     # keep the best checkpoint
    metric_for_best_model="accuracy",
    logging_steps=50,                # print loss every 50 steps
    fp16=torch.cuda.is_available(),  # use mixed precision on GPU
    report_to="none",                # disable wandb/tensorboard
    seed=SEED,
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Create Trainer and start training
# ──────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("=" * 60)
print("STARTING TRAINING")
print(f"  Epochs:         {NUM_EPOCHS}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Learning rate:  {LEARNING_RATE}")
print(f"  Optimizer:      AdamW (gradient descent)")
print(f"  Loss function:  CrossEntropyLoss")
print("=" * 60 + "\n")

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Evaluate on test set
# ──────────────────────────────────────────────────────────────────────────────
print("\nFinal evaluation on test set...")
results = trainer.evaluate()
print(f"  Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"  Test F1 Score: {results['eval_f1']:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Save the trained model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)

# Also save the tokenizer so chatbot.py can load it later
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
