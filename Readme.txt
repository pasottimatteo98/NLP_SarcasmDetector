# Sarcasm Detection on Reddit: Comparing Pre-trained Models

This repository contains the implementation and findings of a comparative study on transformer-based models for sarcasm detection and sentiment analysis on Reddit comments.

## Overview

This project examines the effectiveness of four transformer-based models (BERT, RoBERTa, DistilBERT, and ALBERT) in detecting sarcasm in Reddit comments, as well as analyzing the interplay between sarcasm and sentiment. The models were fine-tuned on a balanced dataset of labeled comments and evaluated using various metrics.

## Key Findings

### Sarcasm Detection
- All four models performed well, with accuracies ranging from 0.77 to 0.79
- BERT achieved the highest performance (accuracy: 0.79, F1 score: 0.79)
- DistilBERT followed closely (accuracy: 0.78, F1 score: 0.77)
- RoBERTa and ALBERT demonstrated comparable precision but lower recall

### Sentiment Analysis
- Two approaches were compared: a RoBERTa-based model and VADER
- Significant differences were observed in their classification patterns
- Both models showed limitations in accurately handling sarcastic content
- The models often agreed on neutral sentiment but diverged on positive and negative classifications

## Dataset

The dataset consists of Reddit comments labeled for sarcasm, with the following characteristics:
- Balanced distribution: 54.16% sarcastic and 45.84% non-sarcastic comments
- All comments containing the marker "/s" were labeled as sarcastic (marker was removed for training)
- Extensive preprocessing was applied to clean and standardize the text

## Models

### Sarcasm Detection Models
1. **BERT** (`bert-base-uncased`)
2. **RoBERTa** (`roberta-base`)
3. **ALBERT** (`albert-base-v2`)
4. **DistilBERT** (`distilbert-base-uncased`)

### Sentiment Analysis Models
1. RoBERTa-based model (`cardiffnlp/twitter-roberta-base-sentiment`)
2. VADER (Valence Aware Dictionary and sEntiment Reasoner)

## Methodology

- Models were fine-tuned on the preprocessed dataset
- Training parameters:
  - AdamW optimizer (learning rate: 2e-5, epsilon: 1e-8)
  - 5 epochs with batch size of 32
  - Maximum sequence lengths: 256 tokens for BERT and ALBERT, 128 for RoBERTa and DistilBERT

## Evaluation

Models were evaluated using:
- Accuracy, precision, recall, and F1 score
- ROC curves and AUC
- Confusion matrices
- Qualitative analysis of predictions on example sentences

## Challenges & Future Work

The study highlighted persistent challenges in automated sarcasm detection:
- Context-dependent nature of sarcasm
- Requirement for broader cultural knowledge
- Subtle linguistic cues that can be difficult to detect

Future work could focus on:
- Incorporating broader contextual information (e.g., entire conversation threads)
- Considering subreddit-specific context
- Fine-tuning on more diverse datasets with various types of sarcasm

## Repository Structure

```
├── data/
│   └── sarcasm_on_reddit.csv  # Original dataset (not included - download from Kaggle)
├── models/
│   ├── bert/                  # Fine-tuned BERT model
│   ├── roberta/               # Fine-tuned RoBERTa model
│   ├── albert/                # Fine-tuned ALBERT model
│   └── distilbert/            # Fine-tuned DistilBERT model
├── notebooks/
│   ├── data_preprocessing.ipynb    # Data cleaning and preprocessing
│   ├── sarcasm_detection.ipynb     # Model training and evaluation
│   └── sentiment_analysis.ipynb    # Sentiment analysis experiments
├── src/
│   ├── preprocessing.py       # Preprocessing functions
│   ├── model_utils.py         # Utility functions for models
│   ├── evaluation.py          # Evaluation metrics and visualization
│   └── sentiment.py           # Sentiment analysis implementation
├── results/
│   ├── figures/               # Generated plots and visualizations
│   └── metrics/               # Performance metrics
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- Hutto, C.J., & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.

## Authors

- Matteo Pasotti
- Alexandre Crivellari
- Andrea Muscio

## Repository Link

GitHub: [https://github.com/alexcri90/NLP_SarcasmDetector_Bicocca](https://github.com/alexcri90/NLP_SarcasmDetector_Bicocca)
