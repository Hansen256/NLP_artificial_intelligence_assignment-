# NLP Extensions: Swahili Sentiment Analysis

A comprehensive natural language processing project demonstrating sentiment analysis on Swahili Twitter data using multiple state-of-the-art techniques.

## Project Overview

This project implements a fully standalone notebook with **6 different modeling approaches** for binary sentiment classification on Swahili text:

- **BiLSTM** - Bidirectional LSTM with custom embeddings
- **GRU** - Bidirectional GRU with custom embeddings  
- **Word2Vec** - Classical static word embeddings with Logistic Regression
- **FastText** - Subword-aware embeddings with Logistic Regression
- **Fine-tuned BERT** - Cross-lingual transfer (Swahili→English translation + English BERT fine-tuning)
- **Comprehensive Evaluation** - Confusion matrices, misclassification analysis, and embedding visualization

## Key Features

### 1. Data Processing

- Dataset: Swahili Twitter sentiment data from Hugging Face
- Preprocessing with Swahili-specific stopwords
- Text cleaning, tokenization, and padding
- 80/20 train-test split with stratification

### 2. Exploratory Data Analysis (EDA)

- Dataset statistics and label distribution
- Text length analysis (original vs cleaned)
- Word clouds by sentiment category
- Comprehensive visualizations

### 3. Model Implementations

- **Recurrent Models (BiLSTM/GRU)**: Custom PyTorch implementations with bidirectional processing and early stopping
- **Static Embeddings (Word2Vec/FastText)**: Gensim implementations with logistic regression classifiers
- **Transformer Models (BERT)**: Cross-lingual transfer pipeline with English BERT fine-tuning on translated text

### 4. Training & Evaluation

- 10-epoch training for BiLSTM/GRU with early stopping (patience=3)
- 3-epoch fine-tuning for BERT with validation monitoring
- Training loss curves for convergence visualization
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices and misclassification analysis

### 5. Cross-Lingual Transfer Pipeline

- **Translation**: Swahili → English using Helsinki-NLP Opus-MT
- **Fine-tuning**: English BERT (bert-base-uncased) on translated text
- **Evaluation**: Per-class metrics and comparative analysis

### 6. Embedding Analysis

- t-SNE and PCA dimensionality reduction
- Semantic clustering visualization
- Embedding space geometry analysis
- Tokenization strategy comparison

## Performance Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| BiLSTM | 47.46% | 0.47 | 0.47 | 0.47 |
| GRU | 55.41% | 0.54 | 0.55 | 0.55 |
| Word2Vec | 59.16% | 0.35 | 0.59 | 0.44 |
| FastText | 59.16% | 0.35 | 0.59 | 0.44 |
| **Fine-tuned BERT** | **57.84%** | **0.52** | **0.58** | **0.49** |

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone or download this repository:

```bash
cd NLP_assignment
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. For translation support, you may need Google Cloud credentials:

```bash
#
 Set up Google Cloud service account (optional, fallback to googletrans available)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Usage

### Running the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook NLP_Extensions.ipynb
```

The notebook is fully self-contained and includes:

1. Automatic package installation
2. Data loading and preprocessing
3. Model training and evaluation
4. Result visualization and comparison

### Key Sections

- **Section 0**: Data loading & preprocessing (Swahili Twitter dataset)
- **Section 0.1**: Exploratory Data Analysis with visualizations
- **Section 1**: BiLSTM model implementation and training
- **Section 2**: GRU model implementation and training
- **Section 3**: Word2Vec embeddings with Logistic Regression
- **Section 4**: FastText embeddings with Logistic Regression
- **Section 5**: Cross-lingual transfer (Translation + BERT fine-tuning)
- **Section 6**: Comprehensive model comparison and visualization
- **Section 7**: Tokenization strategies and embedding analysis

## Dataset

**Source**: Swahili Twitter Sentiment Dataset from Hugging Face Hub

- **Original classes**: 3 classes (negative, neutral, positive)
- **Converted to**: Binary classification (negative vs positive)
- **Train set**: ~80% of data
- **Test set**: ~20% of data
- **Language**: Swahili with English translations

## Architecture Details

### BiLSTM/GRU

- Embedding dimension: 64
- Hidden dimension: 32
- Number of layers: 2
- Bidirectional: Yes
- Dropout: 0.3
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss

### Word2Vec/FastText

- Vector size: 64
- Window size: 5
- Minimum count: 1
- Epochs: 5
- Classifier: Logistic Regression (max_iter=1000)

### Fine-tuned BERT

- Base model: bert-base-uncased
- Pre-trained parameters: 110M
- Fine-tuned parameters: 1,536 (classification head only)
- Optimizer: AdamW (lr=2e-5)
- Epochs: 3
- Batch size: 8
- Scheduler: Linear warmup with cosine decay

## Files

- `NLP_Extensions.ipynb` - Main Jupyter notebook with all implementations
- `NLP_Results_Report.md` - Detailed results, analysis, and interpretations
- `requirements.txt` - Python package dependencies
- `README.md` - This file

## References

### Datasets

- [Swahili Tweet Sentiment Dataset](https://huggingface.co/datasets/Davis/Swahili-tweet-sentiment)

**Last Updated**: November 5, 2025  
**Project Status**: Complete and documented  
**Notebook Execution**: Fully standalone with automatic dependency installation
