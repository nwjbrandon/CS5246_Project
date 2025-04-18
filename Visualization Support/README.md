# Visualization support

This folder contains code for simplifying text by reducing complexity while preserving meaning and experimenting with image generation.

## Files Explanation

### Jupyter Notebooks

- **Image generation Notebook** - Shows the building of BERT model and image generation using Stable Diffusion
  - Trains BART over 2 epochs using the CompLex dataset.
  - Makes use of WordNet to derive semantic relationships
  - Use a Stable Diffusion pipeline to generate images
  - Experiments on the effectiveness of image generation for a given prompt

## How to Use

Basic example:

```python
# Identification of most complex word and a gloss
from Image_Generation import get_most_complex_word_gloss
sentence = "The enigmatic phenomenon perplexed the researchers."
print(get_most_complex_word_gloss(sentence, threshold=0.3))

# Identification of complex words for a threshold and it's closest synonym
from Image_Generation import get_synonym_image
sentence = "The enigmatic phenomenon perplexed the researchers."
print(get_synonym_image(sentence, threshold=0.25))
```

## Requirements

### Complex Word Identification

- Python 3.7+
- NLTK
- Pandas
- NumPy
- Scikit-learn

### Stable Diffusion
- Diffusers
- Transformers
- Accelerate
- Ftfy
