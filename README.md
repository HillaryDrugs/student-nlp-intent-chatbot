# Intent Detection Chatbot Using BERT

A university NLP project that fine-tunes BERT to detect user intents from text. Built for the "Building an Intelligent Application" course (Introduction to Large Language Models).

## What it does

Type a sentence, and the chatbot tells you what the user wants. For example, "What's the weather today?" gets classified as `weather`, and "Book a flight to Paris" gets classified as `book_flight`. The model handles 151 different intents.

## Dataset

CLINC150 dataset (Larson et al., 2019) with 23,850 labeled sentences. The dataset is included as `dataset.csv` so the project runs offline. It can also be loaded from HuggingFace (`clinc_oos`).

## Model

BERT-base-uncased (110M parameters) with a classification head (768 -> 151). Fine-tuned for 3 epochs with AdamW optimizer and CrossEntropyLoss.

## Results

- Test accuracy: 86.3%
- Test F1 score: 85.4%

## Project structure

```
student-nlp-intent-chatbot/
├── dataset.csv           # Full dataset (23,850 samples)
├── dataset_loader.py     # Loads data from CSV or HuggingFace
├── train.py              # Fine-tunes BERT (3 epochs)
├── evaluate.py           # Prints accuracy and classification report
├── chatbot.py            # Interactive terminal chatbot
├── classification_report.txt  # Full per-class metrics
├── requirements.txt      # Dependencies
├── README.md             # This file
└── report_template.md    # Project report
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (takes ~5 min on GPU)
python train.py

# 3. Evaluate
python evaluate.py

# 4. Run the chatbot
python chatbot.py
```

## Example

```
You: What is the weather like in Istanbul?
Intent:     weather
Confidence: 66.5%

You: I want to book a flight to London
Intent:     book_flight
Confidence: 77.2%

You: How do I change my PIN?
Intent:     pin_change
Confidence: 58.8%
```

## Technologies

Python 3.11, PyTorch, HuggingFace Transformers, HuggingFace Datasets, scikit-learn, pandas

## References

1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL-HLT.
2. Larson et al. (2019). An Evaluation Dataset for Intent Classification. EMNLP.
3. https://huggingface.co/datasets/clinc_oos
