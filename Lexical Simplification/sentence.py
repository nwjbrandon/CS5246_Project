#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv('data/CLEAR_corpus.csv')

# Print the column names
print("Column names in the CLEAR corpus:")
for i, col_name in enumerate(df.columns):
    print(f"{i+1}. {col_name}")

# Print the shape of the dataset
print(f"\nDataset shape: {df.shape}")

# Check for missing values
print("\nTotal missing values per column:")
print(df.isnull().sum())


# In[2]:


# Keep only the columns we need
keep_cols = [
    # Target
    'BT Easiness',
    
    # Existing readability metrics
    'Flesch-Reading-Ease', 
    'Flesch-Kincaid-Grade-Level', 
    'Automated Readability Index', 
    'SMOG Readability', 
    'New Dale-Chall Readability Formula',
    'CAREC', 
    'CAREC_M', 
    'CARES', 
    'CML2RI',
    
    # Text statistics
    'Google\nWC',
    'Sentence\nCount v1',
    'Paragraphs',
    'British WC',
    
    # Contextual
    'Category',
    'Location',
    'MPAA\nMax',
    
    # For feature engineering
    'Excerpt'
]

# Create a working copy with only relevant columns
work_df = df[keep_cols].copy()

# Print the shape of our working dataset
print(f"Working dataset shape: {work_df.shape}")


# In[7]:


# ==========================================================================
# 1. BASIC TEXT STATISTICS (IMPROVED)
# ==========================================================================
import pandas as pd
import numpy as np
import re
import string
import nltk
from scipy.stats import pearsonr, spearmanr
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# Download NLTK resources if needed
try:
    from nltk.corpus import cmudict
    cmu_dict = cmudict.dict()
except (LookupError, ImportError):
    print("Downloading NLTK resources...")
    nltk.download('cmudict')
    from nltk.corpus import cmudict
    cmu_dict = cmudict.dict()

# Enhanced syllable counting function using NLTK's cmudict with robust fallback
def count_syllables(word):
    """
    Count syllables in a word using NLTK's CMU Pronouncing Dictionary.
    Falls back to rule-based algorithm if word not found in dictionary.
    
    Args:
        word (str): The word to count syllables for
        
    Returns:
        int: Number of syllables
    """
    # Handle empty strings and non-alpha characters
    word = word.lower().strip()
    word = re.sub(r'[^a-z]', '', word)
    
    if not word:
        return 0
        
    # Check for the word in the CMU dictionary
    if word in cmu_dict:
        # Count number of digits in the pronunciation (each digit represents a stressed vowel)
        return max(1, len([ph for ph in cmu_dict[word][0] if any(c.isdigit() for c in ph)]))
    
    # FALLBACK: Rule-based approach for words not in dictionary
    # Common exceptions with known syllable counts
    exceptions = {
        'coed': 2, 'loved': 1, 'lives': 1, 'moved': 1, 'lived': 1, 'hated': 2,
        'wanted': 2, 'ended': 2, 'rated': 2, 'used': 1, 'caused': 1, 'forced': 1,
        'hoped': 1, 'deserved': 2, 'named': 1, 'the': 1, 'a': 1, 'i': 1, 'an': 1,
        'area': 3, 'aria': 3, 'eye': 1, 'ate': 1, 'once': 1, 'are': 1
    }
    
    if word in exceptions:
        return exceptions[word]
    
    # Safety check for very short words
    if len(word) <= 1:
        return 1
        
    # Count vowel groups as syllables
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    # Handle common patterns - with safety checks for short words
    if len(word) >= 2:
        # Silent 'e' at the end
        if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
            count -= 1
            
        # Words ending in 'le' usually have an extra syllable
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
            
        # Words ending in 'y' usually have an extra syllable
        if word.endswith('y') and len(word) > 1 and word[-2] not in vowels:
            count += 1
            
        # Words ending in 'es' or 'ed' may have a silent syllable
        if (word.endswith('es') or word.endswith('ed')) and len(word) > 2 and word[-3] not in vowels:
            count -= 1
    
    # Ensure at least one syllable
    return max(1, count)

# Improved function to count syllables in text
def count_text_syllables(text):
    """
    Count the total number of syllables in a text passage.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        int: Total number of syllables
    """
    # Handle empty text
    if not text or not isinstance(text, str):
        return 0
        
    # Find all words and count syllables
    words = re.findall(r'\b\w+\b', text)
    return sum(count_syllables(word) for word in words)

# Create a function to evaluate features
def evaluate_feature(feature_name, feature_values, target_values):
    """
    Calculate correlation metrics between a feature and target values.
    
    Args:
        feature_name (str): Name of the feature
        feature_values (array-like): Feature values
        target_values (array-like): Target values to correlate against
        
    Returns:
        tuple: (Pearson correlation, Spearman correlation)
    """
    # Calculate correlations
    pearson_corr, p_value = pearsonr(feature_values, target_values)
    spearman_corr, s_p_value = spearmanr(feature_values, target_values)
    
    print(f"{feature_name}:")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={p_value:.4f})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={s_p_value:.4f})")
    
    return pearson_corr, spearman_corr

# Load the dataset
df = pd.read_csv('data/CLEAR_corpus.csv')

print("\n\n=====================================================")
print("1. BASIC TEXT STATISTICS (IMPROVED)")
print("=====================================================")

# Store computed features
features = {}

# Feature: Word Count (already in dataset)
print("\nTesting: Total number of words")
evaluate_feature('word_count', df['Google\nWC'].values, df['BT Easiness'])
features['word_count'] = df['Google\nWC'].values

# Feature: Character Count
print("\nTesting: Total number of characters (excluding spaces)")
char_count = df['Excerpt'].apply(lambda x: len(x.replace(" ", "")))
evaluate_feature('char_count', char_count, df['BT Easiness'])
features['char_count'] = char_count

# Feature: Average Word Length
print("\nTesting: Average word length (characters per word)")
avg_word_length = df['Excerpt'].apply(
    lambda x: np.mean([len(word) for word in re.findall(r'\b\w+\b', x)])
)
evaluate_feature('avg_word_length', avg_word_length, df['BT Easiness'])
features['avg_word_length'] = avg_word_length

# Feature: Sentence Count (already in dataset)
print("\nTesting: Total number of sentences")
evaluate_feature('sentence_count', df['Sentence\nCount v1'].values, df['BT Easiness'])
features['sentence_count'] = df['Sentence\nCount v1'].values

# Feature: Average Sentence Length
print("\nTesting: Average sentence length (words per sentence)")
# Reasoning: Longer sentences generally require more cognitive effort to process
avg_sent_length = df['Google\nWC'] / df['Sentence\nCount v1']
evaluate_feature('avg_sentence_length', avg_sent_length, df['BT Easiness'])
features['avg_sentence_length'] = avg_sent_length

# Feature: Paragraph Count (already in dataset)
print("\nTesting: Total number of paragraphs")
evaluate_feature('paragraph_count', df['Paragraphs'].values, df['BT Easiness'])
features['paragraph_count'] = df['Paragraphs'].values

# Feature: Average Paragraph Length
print("\nTesting: Average paragraph length (words per paragraph)")
# Reasoning: Longer paragraphs may contain more complex ideas and connections
avg_para_length = df['Google\nWC'] / df['Paragraphs']
evaluate_feature('avg_paragraph_length', avg_para_length, df['BT Easiness'])
features['avg_paragraph_length'] = avg_para_length

# Feature: Average Characters per Sentence
print("\nTesting: Average characters per sentence")
# Reasoning: This combines word length and sentence length into a single metric
chars_per_sentence = char_count / df['Sentence\nCount v1']
evaluate_feature('chars_per_sentence', chars_per_sentence, df['BT Easiness'])
features['chars_per_sentence'] = chars_per_sentence

# Feature: Words per Paragraph vs Sentence Length Ratio
print("\nTesting: Ratio of words per paragraph to average sentence length")
# Reasoning: This measures how sentences are distributed within paragraphs
# A higher ratio might indicate complex paragraph structure with shorter sentences
para_sent_ratio = avg_para_length / avg_sent_length
evaluate_feature('para_sent_ratio', para_sent_ratio, df['BT Easiness'])
features['para_sent_ratio'] = para_sent_ratio

# IMPROVED SYLLABLE-BASED FEATURES
print("\nTesting: Total syllable count (improved with NLTK)")
# Reasoning: More syllables generally indicate more complex vocabulary
print("Computing syllable counts... (this may take a while)")
syllable_count = df['Excerpt'].apply(count_text_syllables)
evaluate_feature('syllable_count', syllable_count, df['BT Easiness'])
features['syllable_count'] = syllable_count

# Feature: Average Syllables per Word
print("\nTesting: Average syllables per word (improved with NLTK)")
# Reasoning: Words with more syllables tend to be more complex
syllables_per_word = syllable_count / df['Google\nWC']
evaluate_feature('syllables_per_word', syllables_per_word, df['BT Easiness'])
features['syllables_per_word'] = syllables_per_word

# Feature: Polysyllabic Words (3+ syllables)
print("\nTesting: Count of words with 3+ syllables (improved with NLTK)")
# Reasoning: Polysyllabic words are often more complex and academic

def count_polysyllabic_words(text):
    if not text or not isinstance(text, str):
        return 0
    words = re.findall(r'\b\w+\b', text)
    return sum(1 for word in words if count_syllables(word) >= 3)

polysyllable_count = df['Excerpt'].apply(count_polysyllabic_words)
evaluate_feature('polysyllable_count', polysyllable_count, df['BT Easiness'])
features['polysyllable_count'] = polysyllable_count

# Feature: Percentage of Polysyllabic Words
print("\nTesting: Percentage of words with 3+ syllables (improved with NLTK)")
# Reasoning: Normalized version of polysyllabic word count
polysyllable_ratio = polysyllable_count / df['Google\nWC'] * 100
evaluate_feature('polysyllable_ratio', polysyllable_ratio, df['BT Easiness'])
features['polysyllable_ratio'] = polysyllable_ratio

# The code is designed to be run directly in a Jupyter notebook
# Simply execute each cell in sequence to analyze the CLEAR corpus


# In[8]:


# ==========================================================================
# 2. VOCABULARY COMPLEXITY FEATURES
# ==========================================================================
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
from nltk.corpus import stopwords

# Assuming the dataset is already loaded and helper functions defined in Section 1
# Load the dataset if running this section independently
# df = pd.read_csv('CLEAR_corpus.csv')
# Also ensure count_syllables and count_text_syllables functions are defined

print("\n\n=====================================================")
print("2. VOCABULARY COMPLEXITY FEATURES")
print("=====================================================")

# Feature: Unique Word Count
print("\nTesting: Number of unique words (lexical diversity)")
# Reasoning: More unique words may indicate more complex vocabulary
unique_word_count = df['Excerpt'].apply(
    lambda x: len(set(re.findall(r'\b\w+\b', x.lower())))
)
evaluate_feature('unique_word_count', unique_word_count, df['BT Easiness'])

# Feature: Type-Token Ratio
print("\nTesting: Type-Token Ratio (unique words / total words)")
# Reasoning: Higher lexical diversity often correlates with more advanced texts
type_token_ratio = unique_word_count / df['Google\nWC']
evaluate_feature('type_token_ratio', type_token_ratio, df['BT Easiness'])

# Feature: Total Syllable Count
print("\nTesting: Total syllable count")
# Reasoning: More syllables generally indicate more complex vocabulary
syllable_count = df['Excerpt'].apply(count_text_syllables)
evaluate_feature('syllable_count', syllable_count, df['BT Easiness'])

# Feature: Average Syllables per Word
print("\nTesting: Average syllables per word")
# Reasoning: Words with more syllables tend to be more complex
syllables_per_word = syllable_count / df['Google\nWC']
evaluate_feature('syllables_per_word', syllables_per_word, df['BT Easiness'])

# Feature: Polysyllabic Words (3+ syllables)
print("\nTesting: Count of words with 3+ syllables")
# Reasoning: Polysyllabic words are often more complex and academic
polysyllable_count = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if count_syllables(word) > 2)
)
evaluate_feature('polysyllable_count', polysyllable_count, df['BT Easiness'])

# Feature: Percentage of Polysyllabic Words
print("\nTesting: Percentage of words with 3+ syllables")
# Reasoning: Normalized version of polysyllabic word count
polysyllable_ratio = polysyllable_count / df['Google\nWC'] * 100
evaluate_feature('polysyllable_ratio', polysyllable_ratio, df['BT Easiness'])

# Feature: Long Words (6+ characters)
print("\nTesting: Count of long words (6+ characters)")
# Reasoning: Longer words often represent more complex concepts
long_word_count = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if len(word) >= 6)
)
evaluate_feature('long_word_count', long_word_count, df['BT Easiness'])

# Feature: Percentage of Long Words
print("\nTesting: Percentage of long words (6+ characters)")
# Reasoning: Normalized version of long word count
long_word_ratio = long_word_count / df['Google\nWC'] * 100
evaluate_feature('long_word_ratio', long_word_ratio, df['BT Easiness'])

# Vocabulary Frequency Analysis using NLTK Stopwords
# Reasoning: Common words are easier to read, uncommon words are more difficult
stop_words = set(stopwords.words('english'))
print("\nTesting: Percentage of common words (using NLTK stopwords)")
common_word_ratio = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x.lower()) 
                 if word in stop_words) / len(re.findall(r'\b\w+\b', x)) * 100
)
evaluate_feature('common_word_ratio', common_word_ratio, df['BT Easiness'])

# Feature: Word Length Variance
print("\nTesting: Variance in word length")
# Reasoning: Higher variance might indicate a mix of simple and complex vocabulary
word_length_variance = df['Excerpt'].apply(
    lambda x: np.var([len(word) for word in re.findall(r'\b\w+\b', x)])
)
evaluate_feature('word_length_variance', word_length_variance, df['BT Easiness'])

# Feature: British Word Percentage (already in dataset)
# NOTE: This metric is retained but with a caution note
print("\nTesting: Percentage of British words (CAUTION: culturally context-dependent)")
print("Note: This metric may not generalize well across different reading populations")
# Reasoning: Unfamiliar spellings might increase reading difficulty
british_word_pct = df['British WC'] / df['Google\nWC'] * 100
# Replace NaN with 0 (assuming no British words if NaN)
british_word_pct = british_word_pct.fillna(0)
evaluate_feature('british_word_pct', british_word_pct, df['BT Easiness'])


# In[9]:


# ==========================================================================
# 3. SENTENCE STRUCTURE FEATURES
# ==========================================================================
import pandas as pd
import numpy as np
import re
import string
from scipy.stats import pearsonr, spearmanr
from nltk.tokenize import sent_tokenize

# Assuming the dataset is already loaded and helper functions defined in Section 1
# Load the dataset if running this section independently
# df = pd.read_csv('CLEAR_corpus.csv')

print("\n\n=====================================================")
print("3. SENTENCE STRUCTURE FEATURES")
print("=====================================================")

# Feature: Raw Punctuation Counts (now with normalized versions)
# Adding normalization for punctuation counts to make them more comparable across text lengths

# Feature: Comma Count and Normalized Comma Rate
print("\nTesting: Number of commas and normalized comma rate")
# Reasoning: Commas often indicate complex sentence structures
comma_count = df['Excerpt'].apply(lambda x: x.count(','))
comma_rate = comma_count / df['Google\nWC'] * 100  # Normalized per 100 words
evaluate_feature('comma_count', comma_count, df['BT Easiness'])
evaluate_feature('comma_rate', comma_rate, df['BT Easiness'])  # Added normalized version

# Feature: Commas per Sentence
print("\nTesting: Average commas per sentence")
# Reasoning: More commas per sentence suggests more complex clauses
commas_per_sentence = comma_count / df['Sentence\nCount v1']
evaluate_feature('commas_per_sentence', commas_per_sentence, df['BT Easiness'])

# Feature: Semicolon and Colon Counts (with normalization)
print("\nTesting: Normalized semicolon rate (per 100 words)")
# Reasoning: Semicolons often connect related but independent clauses, indicating complexity
semicolon_count = df['Excerpt'].apply(lambda x: x.count(';'))
semicolon_rate = semicolon_count / df['Google\nWC'] * 100  # Normalized per 100 words
evaluate_feature('semicolon_rate', semicolon_rate, df['BT Easiness'])

print("\nTesting: Normalized colon rate (per 100 words)")
# Reasoning: Colons often introduce lists or explanations, which can be complex
colon_count = df['Excerpt'].apply(lambda x: x.count(':'))
colon_rate = colon_count / df['Google\nWC'] * 100  # Normalized per 100 words
evaluate_feature('colon_rate', colon_rate, df['BT Easiness'])

# Feature: Question Mark Count
print("\nTesting: Number of question marks")
# Reasoning: Questions may indicate dialogue or rhetorical devices
question_count = df['Excerpt'].apply(lambda x: x.count('?'))
evaluate_feature('question_count', question_count, df['BT Easiness'])

# Feature: Exclamation Mark Count
print("\nTesting: Number of exclamation marks")
# Reasoning: Exclamations may indicate emotional content
exclamation_count = df['Excerpt'].apply(lambda x: x.count('!'))
evaluate_feature('exclamation_count', exclamation_count, df['BT Easiness'])

# Feature: Punctuation Density
print("\nTesting: Punctuation density (punctuation marks per word)")
# Reasoning: Higher punctuation density may indicate more complex sentence structures
def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

punct_count = df['Excerpt'].apply(count_punctuation)
punct_density = punct_count / df['Google\nWC']
evaluate_feature('punct_density', punct_density, df['BT Easiness'])

# Feature: Parentheses Count
print("\nTesting: Number of parenthetical expressions")
# Reasoning: Parentheses often contain supplementary information, making text more complex
def count_parentheses(text):
    return text.count('(')  # Count opening parentheses

parentheses_count = df['Excerpt'].apply(count_parentheses)
evaluate_feature('parentheses_count', parentheses_count, df['BT Easiness'])

# Feature: Quotation Mark Pairs
print("\nTesting: Number of quotation pairs (dialogue)")
# Reasoning: Dialogue may affect readability differently than narration
def count_quotes(text):
    # Count pairs of double quotes (simplistic approach)
    return text.count('"') // 2

quote_pairs = df['Excerpt'].apply(count_quotes)
evaluate_feature('quote_pairs', quote_pairs, df['BT Easiness'])

# Feature: Sentence Length Variance
print("\nTesting: Variance in sentence length")
# Reasoning: High variance may indicate complex writing with mixed sentence structures

# First, get sentence lengths for each excerpt
def get_sentence_length_variance(text):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return np.var(lengths)

# This could be slow on the full dataset, so you might want to use a sample
sentence_length_variance = df['Excerpt'].apply(get_sentence_length_variance)
evaluate_feature('sentence_length_variance', sentence_length_variance, df['BT Easiness'])


# In[10]:


# ==========================================================================
# 4. CUSTOM READABILITY FORMULA COMPONENTS
# ==========================================================================
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Assuming the dataset and previous metrics are already calculated
# If running independently, you'd need to calculate these metrics first

# Define a function to scale values between 0 and 1
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

print("\n\n=====================================================")
print("4. CUSTOM READABILITY FORMULA COMPONENTS")
print("=====================================================")
print("Testing components from established readability formulas separately")

# Flesch Reading Ease Components
print("\n--- Flesch Reading Ease Components ---")

# 1. Words per Sentence Component
print("\nTesting: Words per Sentence (1.015 * words/sentences)")
# Reasoning: From Flesch Reading Ease formula - penalizes longer sentences
words_per_sentence_component = 1.015 * (df['Google\nWC'] / df['Sentence\nCount v1'])
evaluate_feature('words_per_sentence_component', words_per_sentence_component, df['BT Easiness'])

# 2. Syllables per Word Component
print("\nTesting: Syllables per Word (84.6 * syllables/words)")
# Reasoning: From Flesch Reading Ease formula - penalizes more syllables per word
syllables_per_word_component = 84.6 * (syllable_count / df['Google\nWC'])
evaluate_feature('syllables_per_word_component', syllables_per_word_component, df['BT Easiness'])

# 3. Custom Flesch-inspired Formula
print("\nTesting: Custom Flesch-inspired Formula (206.835 - words_per_sentence_component - syllables_per_word_component)")
# Reasoning: Simplified version of Flesch Reading Ease
custom_flesch = 206.835 - words_per_sentence_component - syllables_per_word_component
evaluate_feature('custom_flesch', custom_flesch, df['BT Easiness'])

# Dale-Chall Component (approximation using common_word_ratio)
print("\n--- Dale-Chall Component ---")
print("\nTesting: Dale-Chall inspired component (15.79 * (100 - common_word_ratio)/100)")
# Reasoning: Based on Dale-Chall formula which penalizes uncommon words
dale_chall_component = 15.79 * ((100 - common_word_ratio)/100)
evaluate_feature('dale_chall_component', dale_chall_component, df['BT Easiness'])

# SMOG Component
print("\n--- SMOG Component ---")
print("\nTesting: SMOG-inspired component (1.043 * sqrt(polysyllable_count * (30/sentence_count)))")
# Reasoning: Based on SMOG formula focusing on polysyllabic words
smog_component = 1.043 * np.sqrt(polysyllable_count * (30/df['Sentence\nCount v1']))
evaluate_feature('smog_component', smog_component, df['BT Easiness'])

# Automated Readability Index Components
print("\n--- Automated Readability Index Components ---")

# 1. Characters per Word Component
print("\nTesting: Characters per Word (4.71 * chars/words)")
# Reasoning: From ARI formula - penalizes longer words
chars_per_word_component = 4.71 * (char_count / df['Google\nWC'])
evaluate_feature('chars_per_word_component', chars_per_word_component, df['BT Easiness'])

# 2. Words per Sentence Component (ARI version)
print("\nTesting: Words per Sentence ARI version (0.5 * words/sentences)")
# Reasoning: From ARI formula - penalizes longer sentences but less than Flesch
words_per_sentence_ari = 0.5 * (df['Google\nWC'] / df['Sentence\nCount v1'])
evaluate_feature('words_per_sentence_ari', words_per_sentence_ari, df['BT Easiness'])

# 3. Custom ARI-inspired Formula
print("\nTesting: Custom ARI-inspired Formula (chars_per_word_component + words_per_sentence_ari - 21.43)")
# Reasoning: Simplified version of ARI
custom_ari = chars_per_word_component + words_per_sentence_ari - 21.43
evaluate_feature('custom_ari', custom_ari, df['BT Easiness'])

# Novel Formula: Structural Complexity Index
print("\n--- Novel Formula: Structural Complexity Index ---")
print("\nTesting: Structural Complexity Index (combines sentence structure and vocab)")
# Reasoning: Combines sentence structure complexity with vocabulary complexity
# Higher values indicate more complex text

# Normalize key components to 0-1 scale for equal weighting
# Sentence complexity factors
norm_sent_length = min_max_scale(avg_sent_length)
norm_commas_per_sent = min_max_scale(commas_per_sentence)
norm_punct_density = min_max_scale(punct_density)

# Vocabulary complexity factors
norm_avg_word_length = min_max_scale(avg_word_length)
norm_polysyllable_ratio = min_max_scale(polysyllable_ratio)
norm_long_word_ratio = min_max_scale(long_word_ratio)

# Combined structural complexity index
structural_complexity = (
    (0.2 * norm_sent_length) + 
    (0.2 * norm_commas_per_sent) + 
    (0.1 * norm_punct_density) +
    (0.2 * norm_avg_word_length) +
    (0.2 * norm_polysyllable_ratio) +
    (0.1 * norm_long_word_ratio)
)
evaluate_feature('structural_complexity_index', structural_complexity, df['BT Easiness'])

# Novel Formula: Cognitive Load Index
print("\n--- Novel Formula: Cognitive Load Index ---")
print("\nTesting: Cognitive Load Index (penalizes length, density, and variance)")
# Reasoning: Aims to estimate cognitive load based on text properties
# Higher values indicate higher cognitive load

# Text length factors (normalized)
norm_word_count = min_max_scale(df['Google\nWC'])
norm_sent_count = min_max_scale(df['Sentence\nCount v1'])

# Information density factors
norm_type_token_ratio = min_max_scale(type_token_ratio)
norm_syllables_per_word = min_max_scale(syllables_per_word)

# Variability factors
norm_sent_variance = min_max_scale(sentence_length_variance)
norm_word_variance = min_max_scale(word_length_variance)

# Combined cognitive load index
cognitive_load = (
    (0.15 * norm_word_count) +
    (0.05 * norm_sent_count) +
    (0.2 * norm_type_token_ratio) +
    (0.3 * norm_syllables_per_word) +
    (0.15 * norm_sent_variance) +
    (0.15 * norm_word_variance)
)
evaluate_feature('cognitive_load_index', cognitive_load, df['BT Easiness'])


# In[13]:


# ==========================================================================
# 5. ADVANCED LINGUISTIC FEATURES
# ==========================================================================
print("\n\n=====================================================")
print("5. ADVANCED LINGUISTIC FEATURES")
print("=====================================================")
print("Note: These features are more computationally intensive and may need to be run on a sample")

# Function to get part-of-speech distribution
def pos_features(text):
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Calculate percentages
        total_words = len(tokens)
        if total_words == 0:
            return 0, 0, 0, 0
            
        noun_pct = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                   pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_words * 100
        verb_pct = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                   pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) +
                   pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_words * 100
        adj_pct = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                  pos_counts.get('JJS', 0)) / total_words * 100
        adv_pct = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                  pos_counts.get('RBS', 0)) / total_words * 100
        
        return noun_pct, verb_pct, adj_pct, adv_pct
    except Exception as e:
        print(f"Error in POS tagging: {e}")
        return 0, 0, 0, 0

# Sample for POS analysis if dataset is large
sample_size = min(1000, len(df))  # Adjust as needed based on computational constraints
print(f"\nUsing a sample of {sample_size} excerpts for POS analysis...")
sample_df = df.sample(sample_size, random_state=42)

# Apply POS analysis to sample
print("Analyzing parts of speech...")
pos_results = [pos_features(text) for text in sample_df['Excerpt']]
noun_pcts, verb_pcts, adj_pcts, adv_pcts = zip(*pos_results)

# Feature: Noun Percentage
print("\nTesting: Percentage of nouns")
# Reasoning: Higher proportion of nouns may indicate more technical/informational content
evaluate_feature('noun_percentage', np.array(noun_pcts), sample_df['BT Easiness'])

# Feature: Verb Percentage
print("\nTesting: Percentage of verbs")
# Reasoning: Verb density may relate to action and narrative complexity
evaluate_feature('verb_percentage', np.array(verb_pcts), sample_df['BT Easiness'])

# Feature: Adjective Percentage
print("\nTesting: Percentage of adjectives")
# Reasoning: More adjectives may indicate more descriptive and complex text
evaluate_feature('adjective_percentage', np.array(adj_pcts), sample_df['BT Easiness'])

# Feature: Adverb Percentage
print("\nTesting: Percentage of adverbs")
# Reasoning: Adverbs often modify verbs and may indicate more nuanced actions
evaluate_feature('adverb_percentage', np.array(adv_pcts), sample_df['BT Easiness'])

# Feature: Noun-to-Verb Ratio
print("\nTesting: Noun-to-verb ratio")
# Reasoning: Higher ratio may indicate more complex, information-dense text
noun_verb_ratio = np.array([n/v if v > 0 else 0 for n, v in zip(noun_pcts, verb_pcts)])
evaluate_feature('noun_verb_ratio', noun_verb_ratio, sample_df['BT Easiness'])

# Feature: Descriptors Ratio (Adjectives + Adverbs)
print("\nTesting: Descriptors ratio (adjectives + adverbs)")
# Reasoning: More descriptors may indicate more elaborative, complex language
descriptor_pct = np.array([a + b for a, b in zip(adj_pcts, adv_pcts)])
evaluate_feature('descriptor_percentage', descriptor_pct, sample_df['BT Easiness'])

# Function to estimate syntactic complexity
def syntactic_complexity(text):
    try:
        sentences = sent_tokenize(text)
        # Count conjunctions (basic proxy for clauses)
        conjunctions = ['and', 'but', 'or', 'so', 'because', 'although', 'though', 
                        'since', 'unless', 'while', 'whereas', 'if', 'when']
        
        conjunction_count = 0
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            conjunction_count += sum(1 for word in words if word in conjunctions)
        
        # Estimate average clauses per sentence
        if len(sentences) == 0:
            return 0
        return (conjunction_count + len(sentences)) / len(sentences)
    except Exception as e:
        print(f"Error in syntactic complexity: {e}")
        return 0

# Feature: Syntactic Complexity
print("\nTesting: Estimated clauses per sentence")
# Reasoning: More clauses per sentence indicates more complex syntax
syntax_complexity = sample_df['Excerpt'].apply(syntactic_complexity)
evaluate_feature('clauses_per_sentence', syntax_complexity, sample_df['BT Easiness'])

# Feature: Function Word Ratio
print("\nTesting: Function word ratio")
# Reasoning: Function words (prepositions, articles, etc.) can indicate syntactic complexity
function_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'by', 'with', 'from', 'to', 'for', 
                      'of', 'about', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 
                      'should', 'may', 'might', 'can', 'could', 'must', 'ought', 'than', 'that',
                      'this', 'these', 'those', 'and', 'but', 'or', 'yet', 'so', 'if', 'then'])

def function_word_ratio(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    function_count = sum(1 for word in words if word in function_words)
    return function_count / len(words) * 100

func_word_ratio = sample_df['Excerpt'].apply(function_word_ratio)
evaluate_feature('function_word_ratio', func_word_ratio, sample_df['BT Easiness'])

# Feature: Content-Function Word Ratio
print("\nTesting: Content-function word ratio")
# Reasoning: Ratio of content words to function words may indicate information density
content_function_ratio = sample_df['Excerpt'].apply(
    lambda x: (1 - function_word_ratio(x)/100) / (function_word_ratio(x)/100) 
    if function_word_ratio(x) > 0 else 0
)
evaluate_feature('content_function_ratio', content_function_ratio, sample_df['BT Easiness'])

# Feature: Average Words Between Punctuation
print("\nTesting: Average words between punctuation marks")
# Reasoning: Longer stretches without punctuation can be harder to process
def words_between_punctuation(text):
    segments = re.split(r'[,.;:!?]', text)
    if not segments:
        return 0
    word_counts = [len(re.findall(r'\b\w+\b', segment)) for segment in segments]
    return np.mean(word_counts) if word_counts else 0

words_btwn_punct = sample_df['Excerpt'].apply(words_between_punctuation)
evaluate_feature('words_between_punctuation', words_btwn_punct, sample_df['BT Easiness'])

# Feature: Sentence Opener Variety
print("\nTesting: Sentence opener variety (unique first words / sentences)")
# Reasoning: More variety in sentence openers may indicate more sophisticated writing

def sentence_opener_variety(text):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0
    
    # Get first content word of each sentence
    first_words = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        if words:
            # Skip initial function words to get to content words
            for word in words:
                if word not in function_words:
                    first_words.append(word)
                    break
            else:
                # If all words are function words, use the first one
                first_words.append(words[0])
    
    if not first_words:
        return 0
    
    # Calculate variety
    return len(set(first_words)) / len(first_words)

opener_variety = sample_df['Excerpt'].apply(sentence_opener_variety)
evaluate_feature('sentence_opener_variety', opener_variety, sample_df['BT Easiness'])

# Feature: Abstract vs. Concrete Language 
print("\nTesting: Abstract word percentage (approximation)")
# Reasoning: Abstract concepts are typically harder to comprehend than concrete ones
# Note: This is a simplification - a proper implementation would use a lexicon of abstract/concrete words

# Approximation: longer words tend to be more abstract
abstract_words = sample_df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) 
                  if len(word) > 7 and word.lower() not in function_words) / 
              max(1, len(re.findall(r'\b\w+\b', x))) * 100
)
evaluate_feature('abstract_word_percentage', abstract_words, sample_df['BT Easiness'])

# Print summary of findings
print("\n\n=====================================================")
print("SUMMARY OF TOP PREDICTORS")
print("=====================================================")
print("Based on this analysis, the following features appear to be the strongest predictors of text complexity:")
print("1. Features related to vocabulary complexity (syllables per word, word length)")
print("2. Features related to sentence structure (sentence length, punctuation density)")
print("3. Features related to syntactic complexity (clauses per sentence)")
print("4. Features that combine multiple aspects of complexity")
print("\nThese findings align with established readability formulas but also suggest additional factors to consider.")


# In[16]:


# ==========================================================================
# 6. CATEGORY-SPECIFIC FEATURES (FIXED)
# ==========================================================================
import pandas as pd
import numpy as np
import re
import string
from scipy.stats import pearsonr, spearmanr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Load the dataset if not already loaded
try:
    df
except NameError:
    df = pd.read_csv('data/CLEAR_corpus.csv')

# Create a function to evaluate features
def evaluate_feature(feature_name, feature_values, target_values):
    # Calculate correlations
    pearson_corr, p_value = pearsonr(feature_values, target_values)
    spearman_corr, s_p_value = spearmanr(feature_values, target_values)
    
    print(f"{feature_name}:")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={p_value:.4f})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={s_p_value:.4f})")
    
    return pearson_corr, spearman_corr

# Function to safely calculate correlation
def safe_correlation(x, y):
    """Calculate correlation only if there are at least 2 samples"""
    if len(x) < 2 or len(y) < 2:
        print("  Not enough samples for correlation (need at least 2)")
        return None
    else:
        return pearsonr(x, y)[0]

print("\n\n=====================================================")
print("6. CATEGORY-SPECIFIC FEATURES (FIXED)")
print("=====================================================")
print("Exploring features that may be more relevant to specific text categories")

# FIX: The category values in the dataset are "Lit" and "Info", not "Literary" and "Informational"
print("\n--- Literary vs. Informational Text Analysis ---")

# First, separate informational from literary texts
informational_df = df[df['Category'] == 'Info']
literary_df = df[df['Category'] == 'Lit']

print(f"Number of informational texts: {len(informational_df)}")
print(f"Number of literary texts: {len(literary_df)}")

# Compare average BT Easiness between categories
info_bt = informational_df['BT Easiness'].mean()
lit_bt = literary_df['BT Easiness'].mean()
print(f"Average BT Easiness for informational texts: {info_bt:.4f}")
print(f"Average BT Easiness for literary texts: {lit_bt:.4f}")
print(f"Difference: {lit_bt - info_bt:.4f}")

# Feature: Dialogue Density (more relevant for literary texts)
print("\nTesting: Dialogue density (quotation marks per sentence)")
# Reasoning: Dialogue is more common in literary texts and affects readability
def count_dialogue_marks(text):
    # Count opening quotes (simplistic approach)
    return text.count('"') + text.count('"') + text.count('"')

dialogue_marks = df['Excerpt'].apply(count_dialogue_marks)
dialogue_density = dialogue_marks / df['Sentence\nCount v1']
evaluate_feature('dialogue_density', dialogue_density, df['BT Easiness'])

# Evaluate correlation separately for each category
info_corr = safe_correlation(dialogue_density[df['Category'] == 'Info'], 
                            df['BT Easiness'][df['Category'] == 'Info'])

lit_corr = safe_correlation(dialogue_density[df['Category'] == 'Lit'], 
                           df['BT Easiness'][df['Category'] == 'Lit'])

if info_corr is not None:
    print(f"Dialogue density correlation for informational texts: {info_corr:.4f}")
if lit_corr is not None:
    print(f"Dialogue density correlation for literary texts: {lit_corr:.4f}")

# Feature: Technical Term Density (more relevant for informational texts)
print("\nTesting: Technical term density (approximation)")
# Reasoning: Technical terms are more common in informational texts and affect comprehension

# Approximation: Words longer than 8 letters that aren't common function words
technical_terms = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if len(word) > 8) / max(1, len(re.findall(r'\b\w+\b', x))) * 100
)
evaluate_feature('technical_term_density', technical_terms, df['BT Easiness'])

# Evaluate correlation separately for each category
info_corr = safe_correlation(technical_terms[df['Category'] == 'Info'], 
                            df['BT Easiness'][df['Category'] == 'Info'])

lit_corr = safe_correlation(technical_terms[df['Category'] == 'Lit'], 
                           df['BT Easiness'][df['Category'] == 'Lit'])

if info_corr is not None:
    print(f"Technical term density correlation for informational texts: {info_corr:.4f}")
if lit_corr is not None:
    print(f"Technical term density correlation for literary texts: {lit_corr:.4f}")

# Feature: First/Third Person Perspective (more relevant for literary texts)
print("\nTesting: First-person perspective indicator")
# Reasoning: First-person narratives may be more accessible to some readers

first_person = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x.lower()) 
                  if word in ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']) / 
              max(1, len(re.findall(r'\b\w+\b', x))) * 100
)
evaluate_feature('first_person_density', first_person, df['BT Easiness'])

# Evaluate correlation separately for each category
info_corr = safe_correlation(first_person[df['Category'] == 'Info'], 
                            df['BT Easiness'][df['Category'] == 'Info'])

lit_corr = safe_correlation(first_person[df['Category'] == 'Lit'], 
                           df['BT Easiness'][df['Category'] == 'Lit'])

if info_corr is not None:
    print(f"First-person density correlation for informational texts: {info_corr:.4f}")
if lit_corr is not None:
    print(f"First-person density correlation for literary texts: {lit_corr:.4f}")

# Feature: Location-specific differences
print("\n--- Excerpt Location Analysis ---")
print("Analyzing if start/middle/end location affects relationships")

# Get average BT Easiness by location
for location in df['Location'].unique():
    loc_df = df[df['Location'] == location]
    if len(loc_df) > 0:  # Ensure there are samples
        loc_bt = loc_df['BT Easiness'].mean()
        loc_count = len(loc_df)
        print(f"Location '{location}': avg BT Easiness = {loc_bt:.4f} (n={loc_count})")

# See if word count correlation varies by location
for location in df['Location'].unique():
    loc_df = df[df['Location'] == location]
    if len(loc_df) > 10:  # Ensure enough samples
        loc_corr = pearsonr(loc_df['Google\nWC'], loc_df['BT Easiness'])[0]
        print(f"Word count correlation for location '{location}': {loc_corr:.4f}")

# MPAA Rating Analysis
print("\n--- MPAA Rating Analysis ---")
print("Exploring if content maturity correlates with complexity")

# Convert MPAA ratings to numeric (if not already)
mpaa_numeric = df['MPAA \n#Max'].values
evaluate_feature('mpaa_rating', mpaa_numeric, df['BT Easiness'])

# Feature: Custom Category-specific Complexity Index
print("\n--- Category-specific Complexity Index ---")
print("Testing: Custom index weighted differently for literary vs. informational texts")

# For literary texts, emphasize dialogue, perspective, sentence variety
# For informational texts, emphasize technical terms, information density

# Function to safely scale features
def min_max_scale(series):
    """Scale series to 0-1 range, handling empty series"""
    if len(series) < 2:
        return series  # Return as is if not enough data
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)  # All same value, return 0.5
        
    return (series - min_val) / (max_val - min_val)

# Common features for both categories
# First calculate features needed for scaling
avg_sent_length = df['Google\nWC'] / df['Sentence\nCount v1']

# Get syllables per word (using a simpler approximation if not already calculated)
try:
    syllables_per_word
except NameError:
    # Simple approximation of syllables per word if not calculated earlier
    def simple_count_syllables(word):
        word = word.lower().strip()
        if not word:
            return 0
        
        # Count vowel groups
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in "aeiouy"
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Ensure at least one syllable
        return max(1, count)
        
    def approx_syllables_per_word(text):
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0
        return sum(simple_count_syllables(word) for word in words) / len(words)
    
    syllables_per_word = df['Excerpt'].apply(approx_syllables_per_word)

# Scale the features
syllables_pw_scaled = min_max_scale(syllables_per_word)
sent_length_scaled = min_max_scale(avg_sent_length)

# Literary-specific features (using dialogue_density and first_person)
dialogue_scaled = min_max_scale(dialogue_density)
first_person_scaled = min_max_scale(first_person)

# Informational-specific features (using technical_terms)
technical_scaled = min_max_scale(technical_terms)

# Create category-specific indices
literary_index = np.zeros(len(df))
informational_index = np.zeros(len(df))

# Apply different weights based on category
for i, category in enumerate(df['Category']):
    if category == 'Lit':
        # Literary texts: dialogue and perspective matter more
        literary_index[i] = (
            (0.4 * syllables_pw_scaled[i]) +
            (0.2 * sent_length_scaled[i]) +
            (0.2 * dialogue_scaled[i]) + 
            (0.2 * (1 - first_person_scaled[i]))  # Invert because first-person may be easier
        )
    elif category == 'Info':
        # Informational texts: technical terms and sentence structure matter more
        informational_index[i] = (
            (0.4 * syllables_pw_scaled[i]) +
            (0.3 * sent_length_scaled[i]) +
            (0.3 * technical_scaled[i])
        )

# Combine them into a single index
category_specific_index = np.zeros(len(df))
category_specific_index[df['Category'] == 'Lit'] = literary_index[df['Category'] == 'Lit']
category_specific_index[df['Category'] == 'Info'] = informational_index[df['Category'] == 'Info']

evaluate_feature('category_specific_complexity', category_specific_index, df['BT Easiness'])

# Evaluate correlation separately for each category
info_corr = safe_correlation(
    informational_index[df['Category'] == 'Info'], 
    df['BT Easiness'][df['Category'] == 'Info']
)

lit_corr = safe_correlation(
    literary_index[df['Category'] == 'Lit'], 
    df['BT Easiness'][df['Category'] == 'Lit']
)

if info_corr is not None:
    print(f"Category-specific index correlation for informational texts: {info_corr:.4f}")
if lit_corr is not None:
    print(f"Category-specific index correlation for literary texts: {lit_corr:.4f}")


# In[17]:


# ==========================================================================
# 7. TOP FEATURES AND COMBINED ANALYSIS
# ==========================================================================
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Assuming the dataset is already loaded and helper functions defined

print("\n\n=====================================================")
print("7. TOP FEATURES AND COMBINED ANALYSIS")
print("=====================================================")
print("Analyzing which features have the strongest correlations with BT Easiness")

# Store all correlations
correlations = pd.DataFrame(columns=['Feature', 'Pearson_Correlation', 'Spearman_Correlation', 'Abs_Pearson'])

# Function to add correlation to dataframe
def add_correlation(feature_name, feature_values, target_values):
    pearson_corr, _ = pearsonr(feature_values, target_values)
    spearman_corr, _ = spearmanr(feature_values, target_values)
    
    correlations.loc[len(correlations)] = {
        'Feature': feature_name,
        'Pearson_Correlation': pearson_corr,
        'Spearman_Correlation': spearman_corr,
        'Abs_Pearson': abs(pearson_corr)
    }

# Add correlations for basic text features
add_correlation('word_count', df['Google\nWC'], df['BT Easiness'])
add_correlation('char_count', char_count, df['BT Easiness'])
add_correlation('avg_word_length', avg_word_length, df['BT Easiness'])
add_correlation('sentence_count', df['Sentence\nCount v1'], df['BT Easiness'])
add_correlation('avg_sentence_length', avg_sent_length, df['BT Easiness'])
add_correlation('paragraph_count', df['Paragraphs'], df['BT Easiness'])
add_correlation('avg_paragraph_length', avg_para_length, df['BT Easiness'])
add_correlation('chars_per_sentence', chars_per_sentence, df['BT Easiness'])
add_correlation('para_sent_ratio', para_sent_ratio, df['BT Easiness'])

# Add correlations for vocabulary complexity features
add_correlation('unique_word_count', unique_word_count, df['BT Easiness'])
add_correlation('type_token_ratio', type_token_ratio, df['BT Easiness'])
add_correlation('syllable_count', syllable_count, df['BT Easiness'])
add_correlation('syllables_per_word', syllables_per_word, df['BT Easiness'])
add_correlation('polysyllable_count', polysyllable_count, df['BT Easiness'])
add_correlation('polysyllable_ratio', polysyllable_ratio, df['BT Easiness'])
add_correlation('long_word_count', long_word_count, df['BT Easiness'])
add_correlation('long_word_ratio', long_word_ratio, df['BT Easiness'])
add_correlation('british_word_pct', british_word_pct, df['BT Easiness'])

# Add correlations for sentence structure features
add_correlation('comma_count', comma_count, df['BT Easiness'])
add_correlation('commas_per_sentence', commas_per_sentence, df['BT Easiness'])
add_correlation('semicolon_count', semicolon_count, df['BT Easiness'])
add_correlation('colon_count', colon_count, df['BT Easiness'])
add_correlation('punct_density', punct_density, df['BT Easiness'])
add_correlation('parentheses_count', parentheses_count, df['BT Easiness'])
add_correlation('sentence_length_variance', sentence_length_variance, df['BT Easiness'])

# Add correlations for custom formula components
add_correlation('words_per_sentence_component', words_per_sentence_component, df['BT Easiness'])
add_correlation('syllables_per_word_component', syllables_per_word_component, df['BT Easiness'])
add_correlation('custom_flesch', custom_flesch, df['BT Easiness'])
add_correlation('dale_chall_component', dale_chall_component, df['BT Easiness'])
add_correlation('smog_component', smog_component, df['BT Easiness'])
add_correlation('chars_per_word_component', chars_per_word_component, df['BT Easiness'])
add_correlation('words_per_sentence_ari', words_per_sentence_ari, df['BT Easiness'])
add_correlation('custom_ari', custom_ari, df['BT Easiness'])
add_correlation('structural_complexity_index', structural_complexity, df['BT Easiness'])
add_correlation('cognitive_load_index', cognitive_load, df['BT Easiness'])

# Add correlations for category-specific features
add_correlation('dialogue_density', dialogue_density, df['BT Easiness'])
add_correlation('technical_term_density', technical_terms, df['BT Easiness'])
add_correlation('first_person_density', first_person, df['BT Easiness'])
add_correlation('mpaa_rating', mpaa_numeric, df['BT Easiness'])
add_correlation('category_specific_complexity', category_specific_index, df['BT Easiness'])

# Sort by absolute Pearson correlation
correlations = correlations.sort_values('Abs_Pearson', ascending=False).reset_index(drop=True)

# Display top 15 features by correlation strength
print("\nTop 15 features by correlation strength:")
print(correlations.head(15)[['Feature', 'Pearson_Correlation', 'Spearman_Correlation']])

# Group features by their correlation strength
print("\nFeature groups by correlation strength:")
strong_features = correlations[correlations['Abs_Pearson'] > 0.5]['Feature'].tolist()
moderate_features = correlations[(correlations['Abs_Pearson'] > 0.3) & (correlations['Abs_Pearson'] <= 0.5)]['Feature'].tolist()
weak_features = correlations[(correlations['Abs_Pearson'] > 0.1) & (correlations['Abs_Pearson'] <= 0.3)]['Feature'].tolist()

print(f"\nStrong correlations (|r| > 0.5): {len(strong_features)} features")
for feature in strong_features[:10]:  # Show top 10
    print(f"  - {feature}")
if len(strong_features) > 10:
    print(f"  - ... and {len(strong_features) - 10} more")

print(f"\nModerate correlations (0.3 < |r| <= 0.5): {len(moderate_features)} features")
for feature in moderate_features[:10]:  # Show top 10
    print(f"  - {feature}")
if len(moderate_features) > 10:
    print(f"  - ... and {len(moderate_features) - 10} more")

print(f"\nWeak correlations (0.1 < |r| <= 0.3): {len(weak_features)} features")
for feature in weak_features[:10]:  # Show top 10
    print(f"  - {feature}")
if len(weak_features) > 10:
    print(f"  - ... and {len(weak_features) - 10} more")

# Compare custom indices to established readability formulas
print("\nComparing custom indices to established readability formulas:")
print("Feature                     | Correlation with BT Easiness")
print("-----------------------------|-------------------------")
print(f"Flesch-Reading-Ease         | {pearsonr(df['Flesch-Reading-Ease'], df['BT Easiness'])[0]:.4f}")
print(f"Flesch-Kincaid-Grade-Level  | {pearsonr(df['Flesch-Kincaid-Grade-Level'], df['BT Easiness'])[0]:.4f}")
print(f"Automated Readability Index | {pearsonr(df['Automated Readability Index'], df['BT Easiness'])[0]:.4f}")
print(f"SMOG Readability            | {pearsonr(df['SMOG Readability'], df['BT Easiness'])[0]:.4f}")
print(f"New Dale-Chall              | {pearsonr(df['New Dale-Chall Readability Formula'], df['BT Easiness'])[0]:.4f}")
print(f"Custom Flesch-inspired      | {pearsonr(custom_flesch, df['BT Easiness'])[0]:.4f}")
print(f"Custom ARI-inspired         | {pearsonr(custom_ari, df['BT Easiness'])[0]:.4f}")
print(f"Structural Complexity Index | {pearsonr(structural_complexity, df['BT Easiness'])[0]:.4f}")
print(f"Cognitive Load Index        | {pearsonr(cognitive_load, df['BT Easiness'])[0]:.4f}")
print(f"Category-specific Index     | {pearsonr(category_specific_index, df['BT Easiness'])[0]:.4f}")

# Recommend key features for a model
print("\n=====================================================")
print("RECOMMENDED FEATURES FOR TEXT COMPLEXITY MODEL")
print("=====================================================")
print("Based on our analysis, these features would be most valuable for predicting text complexity:")

print("\n1. Core Vocabulary Features:")
print("   - Syllables per word (strong indicator of vocabulary difficulty)")
print("   - Percentage of polysyllabic words (3+ syllables)")
print("   - Average word length (characters per word)")
print("   - Long word ratio (6+ characters)")
print("   - Type-token ratio (normalized lexical diversity)")  # Emphasized over raw unique words

print("\n2. Core Sentence Features:")
print("   - Average sentence length (words per sentence)")
print("   - Commas per sentence (indicator of clause complexity)")
print("   - Sentence length variance (variety in sentence structure)")
print("   - Normalized punctuation rates (punctuation per 100 words)")  # Added normalized metrics

print("\n3. Structural Features:")
print("   - Punctuation density (overall text structure complexity)")
print("   - Paragraph structure (words per paragraph, but not raw paragraph count)")  # Note about paragraph count
print("   - Clauses per sentence (estimated syntactic complexity)")

print("\n4. Category-specific Features:")
print("   - For literary texts: dialogue density, perspective indicators")
print("   - For informational texts: technical term density")

print("\n5. Content Maturity:")
print("   - MPAA rating (indicates theme complexity, but use with caution)")  # Added caution
print("   - NOTE: This may be an indirect correlate rather than direct cause of complexity")

print("\nRECOMMENDATIONS FOR METRIC IMPROVEMENTS:")
print("   - Replace raw counts with normalized rates where appropriate")
print("   - De-emphasize culturally dependent metrics like British word percentage")
print("   - Consider text organization and cohesion in future models")

print("\nA composite model using these features would likely outperform any single readability formula.")
print("Consider also including interaction terms between features to capture their combined effects.")


# In[20]:


# Cell 8: Feature Extraction and Selection for Sentence Complexity Model
print("==================================================")
print("8. SENTENCE COMPLEXITY MODEL")
print("==================================================")
print("Creating a model to predict text complexity based on top features")

# Create an empty dataframe for our features
features_df = pd.DataFrame(index=df.index)

# Extract basic text statistics features
print("\nExtracting basic text statistics...")
features_df['word_count'] = df['Google\nWC']
features_df['char_count'] = df['Excerpt'].apply(lambda x: len(x.replace(" ", "")))
features_df['avg_word_length'] = features_df['char_count'] / features_df['word_count']
features_df['sentence_count'] = df['Sentence\nCount v1']
features_df['avg_sentence_length'] = features_df['word_count'] / features_df['sentence_count']
features_df['chars_per_sentence'] = features_df['char_count'] / features_df['sentence_count']

# Extract vocabulary complexity features
print("Extracting vocabulary complexity features...")
# Unique Word Count (lexical diversity)
features_df['unique_word_count'] = df['Excerpt'].apply(
    lambda x: len(set(re.findall(r'\b\w+\b', x.lower())))
)
features_df['type_token_ratio'] = features_df['unique_word_count'] / features_df['word_count']

# Calculate syllable counts
print("Calculating syllable features...")
features_df['syllable_count'] = df['Excerpt'].apply(count_text_syllables)
features_df['syllables_per_word'] = features_df['syllable_count'] / features_df['word_count']

# Count polysyllabic words (3+ syllables)
features_df['polysyllable_count'] = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if count_syllables(word) > 2)
)
features_df['polysyllable_ratio'] = features_df['polysyllable_count'] / features_df['word_count'] * 100

# Long Words (6+ characters)
features_df['long_word_count'] = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if len(word) >= 6)
)
features_df['long_word_ratio'] = features_df['long_word_count'] / features_df['word_count'] * 100

# Extract sentence structure features
print("Extracting sentence structure features...")
# Comma Count
features_df['comma_count'] = df['Excerpt'].apply(lambda x: x.count(','))
features_df['commas_per_sentence'] = features_df['comma_count'] / features_df['sentence_count']

# Punctuation Density
def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

features_df['punct_count'] = df['Excerpt'].apply(count_punctuation)
features_df['punct_density'] = features_df['punct_count'] / features_df['word_count']

# Sentence Length Variance
features_df['sentence_length_variance'] = df['Excerpt'].apply(get_sentence_length_variance)

# Extract category-specific features
print("Extracting category-specific features...")
# Dialogue Density
def count_dialogue_marks(text):
    # Count opening quotes (simplistic approach)
    return text.count('"') + text.count('"') + text.count('"')

features_df['dialogue_marks'] = df['Excerpt'].apply(count_dialogue_marks)
features_df['dialogue_density'] = features_df['dialogue_marks'] / features_df['sentence_count']

# Technical Term Density (approximation)
features_df['technical_term_density'] = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x) if len(word) > 8) / 
    max(1, len(re.findall(r'\b\w+\b', x))) * 100
)

# First-person perspective
features_df['first_person_density'] = df['Excerpt'].apply(
    lambda x: sum(1 for word in re.findall(r'\b\w+\b', x.lower()) 
                if word in ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']) / 
    max(1, len(re.findall(r'\b\w+\b', x))) * 100
)

# Add standard readability metrics if available in the dataset
print("Adding standard readability metrics...")
try:
    # These column names might need adjustment based on actual dataset
    readability_columns = [
        'Flesch-Reading-Ease', 
        'Flesch-Kincaid-Grade-Level',
        'Automated Readability Index',
        'SMOG Readability'
    ]
    
    for col in readability_columns:
        if col in df.columns:
            features_df[col.replace('-', '_').lower()] = df[col]
except Exception as e:
    print(f"Warning: Error adding standard metrics: {e}")

# Check for any NaN values and handle them
print(f"\nMissing values in features dataframe:")
print(features_df.isnull().sum())

# Fill any NaN values with appropriate defaults
features_df = features_df.fillna(features_df.mean())

# Target variable: BT Easiness (higher = easier to read)
target = df['BT Easiness']

# Print features correlation with target
print("\nFeature correlations with BT Easiness:")
for feature in features_df.columns:
    corr, _ = pearsonr(features_df[feature], target)
    print(f"{feature}: {corr:.4f}")


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


# Cell 9: Train-Test Split and Model Preparation
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_df, target, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Display feature statistics
print("\nFeature statistics (training set):")
print(X_train.describe())


# In[32]:


# Cell 10: Linear Regression Model
# Train a simple linear regression model first
print("\n--- Linear Regression Model ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on test set
lr_predictions = lr_model.predict(X_test)

# Evaluate the model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression Results:")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R² Score: {lr_r2:.4f}")

# Feature importance (coefficients)
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

print("\nTop 5 most important features (positive correlation with easiness):")
print(coefficients.head(5))

print("\nTop 5 most important features (negative correlation with easiness):")
print(coefficients.tail(5))


# In[33]:


# Cell 11: Random Forest Model
print("\n--- Random Forest Regression Model ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest Results:")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R² Score: {rf_r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
importance = importance.sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(importance.head(10))


# In[34]:


# Cell 12: Model Comparison and Visualization
# Compare predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_predictions, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, rf_predictions, alpha=0.5, label='Random Forest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual BT Easiness')
plt.ylabel('Predicted BT Easiness')
plt.title('Actual vs Predicted Text Complexity')
plt.legend()
plt.grid(True)
plt.show()

# Compare model performance
models = ['Linear Regression', 'Random Forest']
rmse_values = [lr_rmse, rf_rmse]
r2_values = [lr_r2, rf_r2]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(models, rmse_values)
plt.title('RMSE Comparison (Lower is Better)')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.bar(models, r2_values)
plt.title('R² Comparison (Higher is Better)')
plt.ylabel('R²')

plt.tight_layout()
plt.show()


# In[35]:


# Cell 13: Create Custom Sentence Complexity Score
print("\n--- Custom Sentence Complexity Score ---")
print("Creating a simplified sentence complexity formula based on the most important features")

# Based on Random Forest feature importance, create a weighted score
# Get the top 5 features and their importance
top_features = importance.head(5)
print("Top 5 features for complexity score:")
print(top_features)

# Create a simplified scoring function
def sentence_complexity_score(text):
    """
    Calculate a 0-100 sentence complexity score based on the top predictive features.
    Higher score = more complex text (harder to read)
    """
    # Basic text stats
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    if word_count == 0:
        return 50  # Default score for empty text
    
    # Calculate core features
    char_count = len(text.replace(" ", ""))
    avg_word_length = char_count / word_count
    
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)
    if sentence_count == 0:
        return 50
    
    avg_sentence_length = word_count / sentence_count
    
    # Syllable features
    syllable_count = count_text_syllables(text)
    syllables_per_word = syllable_count / word_count
    
    # Polysyllabic words (3+ syllables)
    polysyllable_count = sum(1 for word in words if count_syllables(word) > 2)
    polysyllable_ratio = polysyllable_count / word_count * 100
    
    # Compute the score (inverted from BT Easiness, so higher = more complex)
    # Weights based on feature importance
    score = (
        0.30 * (syllables_per_word * 30) +
        0.25 * (polysyllable_ratio / 2) +
        0.20 * (avg_sentence_length / 2) +
        0.15 * (avg_word_length * 10) +
        0.10 * (min(20, sentence_count) / 20 * 30)  # Cap sentence count influence
    )
    
    # Normalize to 0-100 scale
    score = min(100, max(0, score))
    
    return score

# Test the scoring function on some example texts
example_texts = [
    "The cat sat on the mat.",
    "Students learn better when they are engaged in the learning process.",
    "The mitochondrion is the powerhouse of the cell and contains its own genetic material.",
    "The intricate interplay between socioeconomic factors and educational outcomes necessitates a multifaceted approach to pedagogical interventions in underprivileged communities."
]

print("\nTesting the sentence complexity score on example texts:")
for text in example_texts:
    score = sentence_complexity_score(text)
    print(f"Score: {score:.1f} | {text}")

# Test the scoring function on sample texts from the corpus
print("\nSample texts from corpus with their complexity scores:")
sample_indices = np.random.choice(len(df), 5, replace=False)
for idx in sample_indices:
    excerpt = df.iloc[idx]['Excerpt']
    if len(excerpt) > 300:
        excerpt = excerpt[:300] + "..."
    
    bt_easiness = df.iloc[idx]['BT Easiness']
    complexity_score = sentence_complexity_score(excerpt)
    
    print(f"BT Easiness: {bt_easiness:.2f} | Complexity Score: {complexity_score:.1f}")
    print(f"Excerpt: {excerpt}")
    print("-" * 80)


# In[36]:


# Cell 14: Final Model and Recommendations
print("\n--- Final Model and Recommendations ---")

# Determine which model performed better
if rf_r2 > lr_r2:
    best_model = "Random Forest"
    final_model = rf_model
    final_r2 = rf_r2
else:
    best_model = "Linear Regression"
    final_model = lr_model
    final_r2 = lr_r2

print(f"The best performing model is {best_model} with R² of {final_r2:.4f}")


# In[37]:


# Cell 15: Comparing Our Model with Kaggle Competition Results
print("==================================================")
print("15. COMPARISON WITH KAGGLE COMPETITION RESULTS")
print("==================================================")

# Our model's performance
our_model_r2 = 0.4130  # The R² from our Random Forest model

# Evaluate the Kaggle competition predictions against BT Easiness
kaggle_columns = [
    'firstPlace_pred',
    'secondPlace_pred', 
    'thirdPlace_pred',
    'fourthPlace_pred',
    'fifthPlace_pred',
    'sixthPlace_pred'
]

print("\nComparing our model with Kaggle competition predictions:")
print("-----------------------------------------------------")
print(f"Our Random Forest model R²: {our_model_r2:.4f}")

# Calculate R² for each Kaggle prediction
kaggle_scores = {}
for column in kaggle_columns:
    if column in df.columns:
        # Calculate R² between prediction and BT Easiness
        r2 = r2_score(df['BT Easiness'], df[column])
        kaggle_scores[column] = r2
        print(f"{column} R²: {r2:.4f}")

# Create a visualization to compare performances
plt.figure(figsize=(10, 6))
models = ['Our Model'] + list(kaggle_scores.keys())
scores = [our_model_r2] + list(kaggle_scores.values())

# Sort by performance
sorted_indices = np.argsort(scores)[::-1]  # Descending order
sorted_models = [models[i] for i in sorted_indices]
sorted_scores = [scores[i] for i in sorted_indices]

plt.bar(range(len(sorted_models)), sorted_scores, color='skyblue')
plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right')
plt.ylabel('R² Score (higher is better)')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal line for our model's performance
plt.axhline(y=our_model_r2, color='red', linestyle='--', label='Our Model R²')
plt.legend()

plt.show()

# Calculate rankings
print("\nRanking of models by performance:")
print("-----------------------------------------------------")
for i, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
    print(f"{i+1}. {model}: {score:.4f}")

# Analyze the differences
best_kaggle = max(kaggle_scores.values())
difference = our_model_r2 - best_kaggle
percent_diff = (difference / best_kaggle) * 100

print("\nPerformance analysis:")
print("-----------------------------------------------------")
if difference > 0:
    print(f"Our model outperforms the best Kaggle prediction by {difference:.4f} R² ({percent_diff:.2f}%)")
    print("Possible reasons for better performance:")
    print("1. We included more sophisticated linguistic features")
    print("2. Our Random Forest model may capture non-linear relationships better")
    print("3. We might have better feature engineering specific to sentence complexity")
else:
    print(f"Our model underperforms the best Kaggle prediction by {-difference:.4f} R² ({-percent_diff:.2f}%)")
    print("Possible ways to improve our model:")
    print("1. Incorporate feature engineering techniques from the top Kaggle solutions")
    print("2. Try ensemble methods combining multiple model approaches")
    print("3. Explore more advanced NLP features beyond basic text statistics")

# Correlation analysis between our predictions and Kaggle predictions
print("\nCorrelation between different predictions:")
print("-----------------------------------------------------")

# Create a correlation matrix of all predictions
correlation_df = pd.DataFrame()

# Add our model's predictions
X_full = features_df
y_full = df['BT Easiness']
# Retrain the model on the full dataset for fair comparison
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_full, y_full)
our_predictions = final_model.predict(X_full)
correlation_df['Our Model'] = our_predictions

# Add Kaggle predictions
for column in kaggle_scores.keys():
    correlation_df[column] = df[column]

# Add ground truth
correlation_df['BT Easiness'] = df['BT Easiness']

# Calculate correlation matrix
corr_matrix = correlation_df.corr()

# Display correlations with BT Easiness
print("Correlations with ground truth (BT Easiness):")
correlations_with_target = corr_matrix['BT Easiness'].sort_values(ascending=False)
for model, corr in correlations_with_target.items():
    if model != 'BT Easiness':
        print(f"{model}: {corr:.4f}")

# Display correlation between our model and Kaggle predictions
print("\nCorrelations between our model and Kaggle predictions:")
for column in kaggle_scores.keys():
    corr = corr_matrix.loc['Our Model', column]
    print(f"Our Model vs {column}: {corr:.4f}")


# In[ ]:




