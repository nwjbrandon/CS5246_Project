# Text Summarization

This folder contains code for text summarization using various techniques.

## Files Explanation

### Python Files

- **Text summarizer** - Contains architecture for text summarizers
  - Builds 5 models - LSA, Luhn, TextRank, LexRank and BART
  - Defines rouge score and other relevant metrics for evaluation.

### Jupyter Notebooks

- **Text summarizer with quali Notebook** - Builds a text summarizer using the model in the text_summarizer.py file 
  - TTrains the model on XSum and CNN/DailyMail dataset
  - Use Rouge-1, Rouge-2 and Rouge-L scores to evaluate the model on Precision, Recall and F1 score.
  - Additionally performs qualitative evaluation using a sample news artcile
 
- **Traditional Extractive Summarization notebook** - A baseline model performance
  - Uses the 5 models - LSA, Luhn, TextRank, LexRank and BART
  - Analyze performance without any fine-tuning or external training
 
- **TD-IDF-2 Notebook** - Compares BERT and TD-IDF
  - Trains BERT and TD-IDF with a dataset of articles
  - Comparative analysis of keyword extraction based on methodology, contextual understanding, extraction, accuracy, and adaptability

## Requirements

- Python 3.7+
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Torch
- Transformers
- Keybert
- Gensim
