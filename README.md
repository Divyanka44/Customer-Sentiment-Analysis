# Customer Sentiment Analysis

Sentiment classification and prediction on Amazon electronics customer reviews using NLP and classic machine learning models.

## Overview

Online reviews strongly influence purchase decisions, and with hundreds of millions of products available on marketplaces like Amazon, understanding customer sentiment at scale is valuable for manufacturers and sellers alike. This project analyzes a dataset of Amazon electronics reviews to:

- Classify reviews as positive or negative
- Explore common themes and topics in customer feedback
- Compare several machine learning models on sentiment prediction accuracy

## Dataset

Amazon electronic device reviews, sourced from Kaggle (`Amazon_dataset_final.csv`). The dataset is not included in this repo — download it separately and place it in the working directory (see [Setup](#setup)).

## Approach

**1. Text Preprocessing**
- Cleaning (URL removal, punctuation, whitespace, digit stripping)
- Tokenization
- Spelling correction via `autocorrect`
- Lemmatization
- Stop word removal

**2. Sentiment Labeling**
- Sentiment scores computed with VADER (`vaderSentiment`)
- Reviews labeled positive/negative based on compound polarity score

**3. Exploratory Analysis**
- Sentiment distribution across all reviews
- Product-level breakdown of positive review counts
- Word clouds for positive vs. negative reviews and overall most-used terms
- Topic modeling via Latent Dirichlet Allocation (LDA) on a Count Vectorizer term-document matrix

**4. Prediction Models**
Reviews vectorized with TF-IDF / Count Vectorizer (n-grams 1–2), then split 80/20 train-test to train and evaluate:

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | 0.88 | 83.53% |
| **Linear SVM** | **0.89** | **84.47%** |
| Bernoulli Naive Bayes | 0.55 | 70.55% |
| Decision Tree | 0.87 | 81.13% |

**Conclusion:** The Linear SVM model achieved the highest accuracy and ROC AUC among the models tested.

## Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
wordcloud
autocorrect
transformers
tensorflow
lightgbm
nlp
vaderSentiment
scikit-learn
tensorflow_datasets
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Divyanka44/Customer-Sentiment-Analysis.git
   cd Customer-Sentiment-Analysis
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk wordcloud autocorrect transformers tensorflow lightgbm nlp vaderSentiment scikit-learn tensorflow_datasets
   ```
3. Download the Amazon electronics reviews dataset from Kaggle and place it in the expected path (update the file path in the notebook if needed — it currently points to `/content/sample_data/Amazon_dataset_final.csv`, a Colab-specific path).
4. Open and run `Text_mining_Final_Project_V3.ipynb` in Jupyter or Google Colab.

## Usage

The notebook can also be opened directly in Google Colab via the badge at the top of the file. Run cells sequentially — preprocessing and vectorization steps must complete before the modeling sections.

## Repository Structure

```
Customer-Sentiment-Analysis/
└── Text_mining_Final_Project_V3.ipynb   # Main analysis notebook
```

## Author

Divyanka Phadtare
