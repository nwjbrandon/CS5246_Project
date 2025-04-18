# Text Simplification Project

This repository contains code for analyzing and simplifying text by reducing complexity while preserving meaning.

## Files Explanation

### Python Files

- **word_complexity.py** - Contains functions to analyze and score word complexity
  - `WordComplexityPredictor` class calculates how complex a word is
  - `word_complexity_score()` provides a simple interface to score any word from 0-100

- **sentence_complexity.py** - Contains functions to analyze and score sentence complexity
  - `SentenceComplexityPredictor` class evaluates sentence difficulty
  - `sentence_complexity_score()` scores sentence complexity from 0-100
  - Analyzes features like sentence length, syllable count, and readability metrics

- **final.py** - Implements the actual text simplification methods
  - Functions for word substitution (replacing hard words with easier ones)
  - Methods for deletion-based simplification (removing unnecessary parts)
  - Tree search algorithm to find the best simplification approach
  - Interactive simplification tool

### Jupyter Notebooks

- **Word Complexity Notebook** - Shows the research and development of the word complexity model
  - Analyzes what makes words difficult (syllables, length, frequency, etc.)
  - Examines psycholinguistic features (age of acquisition, familiarity)
  - Tests different models to predict word complexity
  - Achieves R² of 0.55 on test data

- **Sentence Complexity Notebook** - Shows the research on sentence complexity
  - Analyzes sentence structure and vocabulary features
  - Tests correlations between features and complexity
  - Evaluates readability metrics
  - Builds a model with R² of 0.41

- **Final Notebook** - Shows the full text simplification system
  - Combines word and sentence complexity analysis
  - Tests different simplification strategies
  - Measures how well meaning is preserved during simplification
  - Includes examples of simplified texts

## How to Use

Basic example:

```python
# Score word complexity
from word_complexity import word_complexity_score
score = word_complexity_score("ubiquitous")
print(f"Complexity: {score}/100")  # Higher = more complex

# Simplify a sentence
from final import simplify_text_combined
text = "The intricate interplay between socioeconomic factors necessitates a multifaceted approach."
result = simplify_text_combined(text)
print(f"Original: {result['original_text']}")
print(f"Simplified: {result['simplified_text']}")
```

## Requirements

- Python 3.7+
- NLTK
- Pandas
- NumPy
- Scikit-learn
