"""
chatbot.py
----------
Interactive terminal chatbot that predicts the intent of user messages
using the fine-tuned BERT model.

Usage:
    python chatbot.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL_DIR = "./saved_model"
MAX_LENGTH = 128

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load the trained model
# ──────────────────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# Load label names from the dataset metadata
dataset = load_dataset("clinc_oos", "plus", split="test")
label_names = dataset.features["intent"].names

print("Model loaded successfully!\n")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Prediction function
# ──────────────────────────────────────────────────────────────────────────────
def predict_intent(text):
    """
    Takes a text string, tokenizes it, feeds it through BERT,
    and returns the predicted intent label and confidence score.
    """
    # Tokenize the input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    # Forward pass (no gradient computation needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)

    intent = label_names[predicted_class.item()]
    score = confidence.item()
    return intent, score


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Interactive chat loop
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 50)
print("  INTENT DETECTION CHATBOT")
print("  Type a question and I'll detect the intent.")
print("  Type 'quit' to exit.")
print("=" * 50)
print()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not user_input:
        print("Please type something.\n")
        continue

    intent, confidence = predict_intent(user_input)

    print(f"Intent:     {intent}")
    print(f"Confidence: {confidence:.1%}")
    print()
