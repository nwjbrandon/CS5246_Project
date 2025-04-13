#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import nltk
from scipy.stats import pearsonr, spearmanr
from nltk.corpus import cmudict

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/cmudict')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('cmudict')
    nltk.download('punkt')

# Load the pronunciation dictionary
prondict = cmudict.dict()

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("Libraries loaded successfully.")

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------
# In[2]:


df = pd.read_csv('data/lcp_single_train.tsv', delimiter='\t')

df[df.token.isna()]


# In[3]:


# Cell 2: Data exploration
print("Examining the dataset structure...")
# Check basic info
print(f"Dataset shape: {df.shape}")
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# Examine missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove rows with missing tokens
df_clean = df.dropna(subset=['token'])
print(f"\nDataset shape after removing rows with missing tokens: {df_clean.shape}")

# Check complexity score distribution
print("\nComplexity score statistics:")
print(df_clean['complexity'].describe())

# Visualize complexity distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['complexity'], bins=30, kde=True)
plt.title('Distribution of Word Complexity Scores')
plt.xlabel('Complexity Score')
plt.ylabel('Count')
plt.show()

# Check corpus distribution
print("\nCorpus distribution:")
print(df_clean['corpus'].value_counts())

# Display a few examples from each corpus
print("\nSample entries from each corpus:")
for corpus in df_clean['corpus'].unique():
    samples = df_clean[df_clean['corpus'] == corpus].sample(min(3, len(df_clean[df_clean['corpus'] == corpus])))
    print(f"\nSamples from {corpus}:")
    for _, row in samples.iterrows():
        print(f"Token: {row['token']}, Complexity: {row['complexity']}")
        print(f"Sentence: {row['sentence']}")
        print("-" * 50)

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------
# In[4]:


# Cell 4: Advanced NLP Feature Engineering

print("Creating advanced NLP features for word complexity prediction...")

# Import additional NLP libraries
import nltk
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources if not already available
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    print("Downloading additional NLTK resources...")
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Create a copy of the cleaned dataframe for feature engineering
df_features = df_clean.copy()

# Add some debug prints to understand the data better
print(f"\nSample of tokens with hyphens:")
hyphen_tokens = df_clean[df_clean['token'].str.contains('-', regex=False)]
print(f"Found {len(hyphen_tokens)} tokens with hyphens")
if len(hyphen_tokens) > 0:
    print(hyphen_tokens['token'].head(5).tolist())

print(f"\nSample of tokens with digits:")
digit_tokens = df_clean[df_clean['token'].str.contains('\d', regex=True)]
print(f"Found {len(digit_tokens)} tokens with digits")
if len(digit_tokens) > 0:
    print(digit_tokens['token'].head(5).tolist())

print(f"\nSample of tokens with punctuation (other than hyphen):")
punct_tokens = df_clean[df_clean['token'].str.contains('[^\w\s-]', regex=True)]
print(f"Found {len(punct_tokens)} tokens with punctuation")
if len(punct_tokens) > 0:
    print(punct_tokens['token'].head(5).tolist())

# Define syllable counting function
def count_syllables(word):
    """
    Count syllables in a word using CMU Pronouncing Dictionary with fallback.
    """
    # Handle empty strings and non-alpha characters
    word = word.lower().strip()
    cleaned_word = re.sub(r'[^a-z]', '', word)
    
    if not cleaned_word:
        return 0
        
    # Check for the word in the CMU dictionary
    if cleaned_word in prondict:
        # Count number of digits in the pronunciation (each digit represents a stressed vowel)
        return max(1, len([ph for ph in prondict[cleaned_word][0] if any(c.isdigit() for c in ph)]))
    
    # FALLBACK: Rule-based approach for words not in dictionary
    # Count vowel groups as syllables
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in cleaned_word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    # Handle common patterns
    if len(cleaned_word) >= 2:
        # Silent 'e' at the end
        if cleaned_word.endswith('e') and len(cleaned_word) > 2 and cleaned_word[-2] not in vowels:
            count -= 1
            
        # Words ending in 'le' usually have an extra syllable
        if cleaned_word.endswith('le') and len(cleaned_word) > 2 and cleaned_word[-3] not in vowels:
            count += 1
    
    # Ensure at least one syllable
    return max(1, count)

# 1. Basic word features
print("\nExtracting basic word features...")
df_features['word_length'] = df_features['token'].apply(len)
df_features['num_syllables'] = df_features['token'].apply(lambda x: count_syllables(x))

# Character-level features
df_features['num_vowels'] = df_features['token'].apply(lambda x: len(re.findall(r'[aeiou]', x.lower())))
df_features['num_consonants'] = df_features['token'].apply(lambda x: len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', x.lower())))
df_features['vowel_ratio'] = df_features['num_vowels'] / df_features['word_length'].map(lambda x: max(1, x))
df_features['consonant_ratio'] = df_features['num_consonants'] / df_features['word_length'].map(lambda x: max(1, x))
df_features['syllables_per_char'] = df_features['num_syllables'] / df_features['word_length'].map(lambda x: max(1, x))

# 2. Morphological features
print("Extracting morphological features...")
# Make hyphen an ordinal feature (count number of hyphens)
df_features['num_hyphens'] = df_features['token'].apply(lambda x: x.count('-'))
df_features['num_digits'] = df_features['token'].apply(lambda x: len(re.findall(r'\d', x)))
df_features['num_puncts'] = df_features['token'].apply(lambda x: len(re.findall(r'[^\w\s-]', x)))

# Case features
df_features['is_capitalized'] = df_features['token'].apply(lambda x: 1 if x[0].isupper() else 0)
df_features['is_all_caps'] = df_features['token'].apply(lambda x: 1 if x.isupper() and len(x) > 1 else 0)

# Check for common affixes
common_prefixes = ['un', 're', 'in', 'im', 'dis', 'en', 'non', 'de', 'over', 'mis', 'sub', 'pre', 'inter', 'fore', 'anti', 'auto', 'bi', 'co', 'ex', 'mid', 'semi', 'under', 'super']
common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ity', 'ment', 'ness', 'ion', 'ation', 'able', 'ible', 'al', 'ial', 'ful', 'ic', 'ical', 'ious', 'ous', 'ive', 'less', 'y']

def count_prefixes(word):
    word = word.lower()
    count = 0
    for prefix in common_prefixes:
        if word.startswith(prefix):
            count += 1
    return count

def count_suffixes(word):
    word = word.lower()
    count = 0
    for suffix in common_suffixes:
        if word.endswith(suffix):
            count += 1
    return count

df_features['num_prefixes'] = df_features['token'].apply(count_prefixes)
df_features['num_suffixes'] = df_features['token'].apply(count_suffixes)

# 3. WordNet features
print("Extracting WordNet features...")
def get_wordnet_features(word):
    """Extract features from WordNet"""
    word = word.lower()
    # Clean the word to improve matching
    clean_word = re.sub(r'[^\w\s]', '', word)
    
    # Get synsets
    synsets = wn.synsets(clean_word)
    
    # Number of meanings (polysemy)
    num_meanings = len(synsets)
    
    # Word familiarity can be approximated by the number of synsets
    # (more common words tend to have more meanings)
    
    # Average word depth in the hierarchy (deeper words are more specific/technical)
    depths = [ss.min_depth() for ss in synsets]
    avg_depth = sum(depths) / len(depths) if depths else 0
    
    # Number of hypernyms (broader terms)
    hypernyms = []
    for ss in synsets:
        hypernyms.extend(ss.hypernyms())
    num_hypernyms = len(hypernyms)
    
    # Number of hyponyms (more specific terms)
    hyponyms = []
    for ss in synsets:
        hyponyms.extend(ss.hyponyms())
    num_hyponyms = len(hyponyms)
    
    return num_meanings, avg_depth, num_hypernyms, num_hyponyms

# Apply WordNet features (with error handling)
wordnet_features = []
for token in df_features['token']:
    try:
        wordnet_features.append(get_wordnet_features(token))
    except Exception:
        # Default values if an error occurs
        wordnet_features.append((0, 0, 0, 0))

# Add WordNet features to dataframe
df_features['num_meanings'] = [f[0] for f in wordnet_features]
df_features['wordnet_depth'] = [f[1] for f in wordnet_features]
df_features['num_hypernyms'] = [f[2] for f in wordnet_features]
df_features['num_hyponyms'] = [f[3] for f in wordnet_features]

# 4. Part of speech features
print("Extracting POS features...")
def get_pos_tag(token, sentence):
    """Get part of speech tag for a token in a sentence"""
    try:
        # Tokenize and tag the sentence
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        
        # Find the token in the tagged list
        token_lower = token.lower()
        for word, tag in tagged:
            if word.lower() == token_lower:
                return tag
        
        # If token not found directly, try partial matching
        for word, tag in tagged:
            if token_lower in word.lower() or word.lower() in token_lower:
                return tag
                
        return 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'

df_features['pos_tag'] = df_features.apply(lambda row: get_pos_tag(row['token'], row['sentence']), axis=1)

# Map POS tags to broader categories for easier modeling
pos_categories = {
    'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
    'VB': 'verb', 'VBD': 'verb', 'VBG': 'verb', 'VBN': 'verb', 'VBP': 'verb', 'VBZ': 'verb',
    'JJ': 'adj', 'JJR': 'adj', 'JJS': 'adj',
    'RB': 'adv', 'RBR': 'adv', 'RBS': 'adv',
    'DT': 'det', 'PDT': 'det', 'WDT': 'det',
    'PRP': 'pron', 'PRP$': 'pron', 'WP': 'pron', 'WP$': 'pron',
    'IN': 'prep', 'TO': 'prep',
    'CC': 'conj',
    'CD': 'num',
    'UH': 'interj',
    'FW': 'foreign'
}

# Group POS tags into categories
df_features['pos_category'] = df_features['pos_tag'].apply(lambda x: pos_categories.get(x, 'other'))

# One-hot encode the POS categories
pos_dummies = pd.get_dummies(df_features['pos_category'], prefix='pos')
df_features = pd.concat([df_features, pos_dummies], axis=1)

# 5. Phonological features
print("Extracting phonological features...")
def count_consonant_clusters(word):
    """Count sequences of 2+ consonants"""
    return len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{2,}', word.lower()))

def count_complex_phonemes(word):
    """Count complex phoneme combinations"""
    complex_phonemes = ['ph', 'th', 'sh', 'ch', 'wh', 'gh', 'ght', 'sch', 'scr', 'squ', 'kn', 'gn', 'ps', 'pn', 'mn', 'rh']
    word = word.lower()
    return sum(1 for phoneme in complex_phonemes if phoneme in word)

df_features['consonant_clusters'] = df_features['token'].apply(count_consonant_clusters)
df_features['num_complex_phonemes'] = df_features['token'].apply(count_complex_phonemes)
df_features['is_polysyllabic'] = df_features['num_syllables'].apply(lambda x: 1 if x > 2 else 0)

# 6. Context features
print("Extracting context features...")
# Position in sentence
def get_position_info(row):
    token = row['token']
    sentence = row['sentence']
    # Find token's position in the sentence
    words = word_tokenize(sentence.lower())
    token_lower = token.lower()
    
    # First try exact match
    if token_lower in words:
        position = words.index(token_lower) + 1
        relative_position = position / len(words)
        return pd.Series([position, relative_position, len(words)], 
                     index=['token_position', 'relative_position', 'sentence_length'])
    
    # Try substring match if exact match fails
    for i, word in enumerate(words):
        if token_lower in word:
            position = i + 1
            relative_position = position / len(words)
            return pd.Series([position, relative_position, len(words)], 
                         index=['token_position', 'relative_position', 'sentence_length'])
    
    # If no match found
    return pd.Series([0, 0, len(words)], 
                 index=['token_position', 'relative_position', 'sentence_length'])

position_features = df_features.apply(get_position_info, axis=1)
df_features = pd.concat([df_features, position_features], axis=1)

# Calculate average word length in sentence
def avg_word_length_in_sentence(sentence):
    words = word_tokenize(sentence)
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

df_features['avg_word_length_in_sentence'] = df_features['sentence'].apply(avg_word_length_in_sentence)
df_features['word_length_vs_avg'] = df_features['word_length'] / df_features['avg_word_length_in_sentence'].map(lambda x: max(1, x))

# Add language model-inspired features
# We can approximate complexity by tracking less common character sequences
def char_bigram_rarity(token):
    # A proxy for how "unusual" the character combinations are in English
    rare_bigrams = ['bk', 'bz', 'cj', 'cp', 'cv', 'cz', 'dg', 'dj', 'dt', 'fh', 'fp', 'fz', 
                   'gj', 'gv', 'gx', 'hg', 'hj', 'hz', 'jb', 'jc', 'jd', 'jf', 'jg', 'jh',
                   'jk', 'jl', 'jm', 'jn', 'jp', 'jq', 'jr', 'js', 'jt', 'jv', 'jw', 'jx', 
                   'jz', 'kq', 'kx', 'kz', 'lj', 'lq', 'lx', 'mx', 'mz', 'pq', 'pv', 'px',
                   'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj', 'qk', 'ql', 'qm',
                   'qn', 'qo', 'qp', 'qr', 'qs', 'qt', 'qv', 'qw', 'qx', 'qy', 'qz', 'sx',
                   'tq', 'tx', 'vb', 'vf', 'vh', 'vj', 'vk', 'vm', 'vp', 'vq', 'vw', 'vx',
                   'wq', 'wv', 'wx', 'xj', 'xk', 'xz', 'yq', 'yv', 'yx', 'zf', 'zr', 'zx']
    
    token_lower = token.lower()
    count = 0
    for i in range(len(token_lower) - 1):
        bigram = token_lower[i:i+2]
        if bigram in rare_bigrams:
            count += 1
    
    return count / max(1, len(token_lower) - 1)

df_features['rare_bigram_ratio'] = df_features['token'].apply(char_bigram_rarity)

# 7. Corpus-specific features
print("Adding corpus-specific features...")
df_features['is_biomed'] = (df_features['corpus'] == 'biomed').astype(int)
df_features['is_europarl'] = (df_features['corpus'] == 'europarl').astype(int)
df_features['is_bible'] = (df_features['corpus'] == 'bible').astype(int)

# Display feature correlations with complexity
print("\nCalculating feature correlations with complexity...")
numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('complexity')  # Remove the target variable

correlations = {}
for feature in numeric_features:
    # Check if the feature is constant
    if df_features[feature].std() == 0:
        print(f"Skipping correlation for {feature} - it's a constant feature")
        correlations[feature] = 0
    else:
        corr, _ = pearsonr(df_features[feature], df_features['complexity'])
        correlations[feature] = corr

# Sort features by absolute correlation
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nFeature correlations with complexity (sorted by strength):")
for feature, corr in sorted_correlations:
    print(f"{feature}: {corr:.4f}")

# Visualize top 10 correlations
plt.figure(figsize=(12, 8))
top_features = [x[0] for x in sorted_correlations[:10]]
top_correlations = [x[1] for x in sorted_correlations[:10]]

sns.barplot(x=top_correlations, y=top_features)
plt.title('Top 10 Feature Correlations with Complexity')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# Print features summary
print("\nFeatures summary:")
print(df_features[numeric_features].describe())

# Save the top features for modeling
top_feature_names = [feature for feature, corr in sorted_correlations[:15]]
print("\nTop 15 features for modeling:")
print(top_feature_names)

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------

# In[5]:

import os
from datasets import load_dataset
import pandas as pd

# Define the target file path
file_path_aoa = "data/aoa.csv"

# Check if the file already exists
if not os.path.isfile(file_path_aoa):
    # Create the target directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load the dataset
    dataset_aoa = load_dataset("StephanAkkerman/English-Age-of-Acquisition")
    
    # Save the dataset to CSV
    dataset_aoa["train"].to_csv(file_path_aoa, index=False)
    print(f"Dataset downloaded and saved to {file_path_aoa}")
else:
    print(f"Dataset already exists at {file_path_aoa}")

# Load the dataset into a DataFrame
df_aoa = pd.read_csv(file_path_aoa)


# In[6]:


# Define the target file path
file_path_concreteness = "data/concreteness.csv"

# Check if the file already exists
if not os.path.isfile(file_path_concreteness):
    # Create the target directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load the dataset
    dataset_concreteness = load_dataset("StephanAkkerman/concreteness-ratings")
    
    # Save the dataset to CSV
    dataset_concreteness["train"].to_csv(file_path_concreteness, index=False)
    print(f"Dataset downloaded and saved to {file_path_concreteness}")
else:
    print(f"Dataset already exists at {file_path_concreteness}")

# Load the dataset into a DataFrame
df_concreteness = pd.read_csv(file_path_concreteness)


# In[7]:


# Define the target file path
file_path_mrc = "data/mrc.csv"

# Check if the file already exists
if not os.path.isfile(file_path_mrc):
    # Create the target directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load the dataset
    dataset_mrc = load_dataset("StephanAkkerman/MRC-psycholinguistic-database")
    
    # Save the dataset to CSV
    dataset_mrc["train"].to_csv(file_path_mrc, index=False)
    print(f"Dataset downloaded and saved to {file_path_mrc}")
else:
    print(f"Dataset already exists at {file_path_mrc}")

# Load the dataset into a DataFrame
df_mrc = pd.read_csv(file_path_mrc)


# In[8]:


# Define the file path
file_path_subtlex = "data/SUBTLEX-US frequency list with PoS and Zipf information.xlsx"

# Load the dataset into a DataFrame
df_subtlex_us = pd.read_excel(file_path_subtlex)


# In[9]:

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------

# Cell 5: Inspect Psycholinguistic Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Print column information for each psycholinguistic dataset
print("Inspecting the structure of psycholinguistic datasets...\n")

# 1. Inspect AoA Dataset
print("=" * 60)
print("AGE OF ACQUISITION (AoA) DATASET")
print("=" * 60)
print(f"Shape: {df_aoa.shape}")
print(f"Columns: {df_aoa.columns.tolist()}")
print("\nSample data:")
print(df_aoa.head(3))
print("\nData types:")
print(df_aoa.dtypes)
print("\nMissing values:")
print(df_aoa.isnull().sum())

# 2. Inspect Concreteness Dataset
print("\n" + "=" * 60)
print("CONCRETENESS DATASET")
print("=" * 60)
print(f"Shape: {df_concreteness.shape}")
print(f"Columns: {df_concreteness.columns.tolist()}")
print("\nSample data:")
print(df_concreteness.head(3))
print("\nData types:")
print(df_concreteness.dtypes)
print("\nMissing values:")
print(df_concreteness.isnull().sum())

# 3. Inspect MRC Dataset
print("\n" + "=" * 60)
print("MRC PSYCHOLINGUISTIC DATASET")
print("=" * 60)
print(f"Shape: {df_mrc.shape}")
print(f"Columns: {df_mrc.columns.tolist()}")
print("\nSample data:")
print(df_mrc.head(3))
print("\nData types:")
print(df_mrc.dtypes)
print("\nMissing values:")
print(df_mrc.isnull().sum())

# 4. Inspect SUBTLEX Dataset
print("\n" + "=" * 60)
print("SUBTLEX WORD FREQUENCY DATASET")
print("=" * 60)
print(f"Shape: {df_subtlex_us.shape}")
print(f"Columns: {df_subtlex_us.columns.tolist()}")
print("\nSample data:")
print(df_subtlex_us.head(3))
print("\nData types:")
print(df_subtlex_us.dtypes)
print("\nMissing values:")
print(df_subtlex_us.isnull().sum())

# 5. Inspect Complexity Dataset (For Reference)
print("\n" + "=" * 60)
print("COMPLEXITY DATASET")
print("=" * 60)
print(f"Shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()}")
print("\nSample data:")
print(df_clean.head(3))
print("\nData types:")
print(df_clean.dtypes)
print("\nMissing values:")
print(df_clean.isnull().sum())

# Check vocabulary overlap between complexity dataset and psycholinguistic datasets
complexity_vocab = set(df_clean['token'].str.lower())
print("\n" + "=" * 60)
print("VOCABULARY OVERLAP ANALYSIS")
print("=" * 60)

# Prepare word columns based on available columns
if 'Word' in df_aoa.columns:
    aoa_vocab = set(df_aoa['Word'].str.lower())
elif 'word' in df_aoa.columns:
    aoa_vocab = set(df_aoa['word'].str.lower())
else:
    aoa_vocab = set()

if 'Word' in df_concreteness.columns:
    concreteness_vocab = set(df_concreteness['Word'].str.lower())
elif 'word' in df_concreteness.columns:
    concreteness_vocab = set(df_concreteness['word'].str.lower())
else:
    concreteness_vocab = set()

# For MRC, we need to check the actual column names
mrc_word_col = [col for col in df_mrc.columns if col.lower() in ['word', 'wrd']]
if mrc_word_col:
    mrc_vocab = set(df_mrc[mrc_word_col[0]].str.lower())
else:
    mrc_vocab = set()

if 'Word' in df_subtlex_us.columns:
    subtlex_vocab = set(df_subtlex_us['Word'].str.lower())
elif 'word' in df_subtlex_us.columns:
    subtlex_vocab = set(df_subtlex_us['word'].str.lower())
else:
    subtlex_vocab = set()

# Calculate overlap percentages
complexity_count = len(complexity_vocab)
aoa_overlap = len(complexity_vocab.intersection(aoa_vocab))
concreteness_overlap = len(complexity_vocab.intersection(concreteness_vocab))
mrc_overlap = len(complexity_vocab.intersection(mrc_vocab))
subtlex_overlap = len(complexity_vocab.intersection(subtlex_vocab))

print(f"Complexity dataset unique words: {complexity_count}")
print(f"AoA overlap: {aoa_overlap} words ({aoa_overlap/complexity_count*100:.2f}%)")
print(f"Concreteness overlap: {concreteness_overlap} words ({concreteness_overlap/complexity_count*100:.2f}%)")
print(f"MRC overlap: {mrc_overlap} words ({mrc_overlap/complexity_count*100:.2f}%)")
print(f"SUBTLEX overlap: {subtlex_overlap} words ({subtlex_overlap/complexity_count*100:.2f}%)")

# Show some examples of words in complexity dataset but not in psycholinguistic datasets
print("\nExample words in complexity dataset not found in psycholinguistic datasets:")
missing_in_all = list(complexity_vocab - (aoa_vocab | concreteness_vocab | mrc_vocab | subtlex_vocab))
if missing_in_all:
    print(", ".join(sorted(missing_in_all[:20])))
    if len(missing_in_all) > 20:
        print(f"...and {len(missing_in_all) - 20} more")
else:
    print("All words in complexity dataset are present in at least one psycholinguistic dataset.")


# In[21]:

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------

# Cell 6: Grouped Psycholinguistic Features Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Create a copy of complexity dataset with lowercase tokens
df_complexity = df_clean.copy()
df_complexity['word_lower'] = df_complexity['token'].str.lower()

print("Analyzing psycholinguistic features by dataset group...")

# =================================================================
# 1. SUBTLEX FEATURES (Word Frequency & Usage)
# =================================================================
print("\n" + "=" * 60)
print("GROUP 1: SUBTLEX FEATURES (100% overlap)")
print("=" * 60)

# Merge complexity with SUBTLEX
df_subtlex_temp = df_subtlex_us.copy()
df_subtlex_temp['word_lower'] = df_subtlex_temp['Word'].str.lower()
df_subtlex_merged = pd.merge(df_complexity, df_subtlex_temp, on='word_lower', how='left')
print(f"Merged shape: {df_subtlex_merged.shape}")

# Create frequency-based features
df_subtlex_merged['freq_log10'] = np.log10(df_subtlex_merged['FREQcount'] + 1)  # Log frequency
df_subtlex_merged['zipf_value'] = df_subtlex_merged['Zipf-value']  # Zipf value (standardized frequency)
df_subtlex_merged['contextual_diversity'] = df_subtlex_merged['SUBTLCD']  # Contextual diversity
df_subtlex_merged['cd_log10'] = np.log10(df_subtlex_merged['CDcount'] + 1)  # Log contextual diversity

# Part of speech features (categorical)
df_subtlex_merged['is_noun'] = (df_subtlex_merged['Dom_PoS_SUBTLEX'] == 'Noun').astype(int)
df_subtlex_merged['is_verb'] = (df_subtlex_merged['Dom_PoS_SUBTLEX'] == 'Verb').astype(int)
df_subtlex_merged['is_adjective'] = (df_subtlex_merged['Dom_PoS_SUBTLEX'] == 'Adjective').astype(int)
df_subtlex_merged['is_adverb'] = (df_subtlex_merged['Dom_PoS_SUBTLEX'] == 'Adverb').astype(int)
df_subtlex_merged['is_function_word'] = (df_subtlex_merged['Dom_PoS_SUBTLEX'].isin(['Article', 'Preposition', 'Pronoun', 'Conjunction'])).astype(int)

# POS dominance (how strongly a word belongs to its primary part of speech)
df_subtlex_merged['pos_dominance'] = df_subtlex_merged['Percentage_dom_PoS']

# Calculate correlations with complexity
subtlex_features = [
    'freq_log10', 'zipf_value', 'contextual_diversity', 'cd_log10',
    'is_noun', 'is_verb', 'is_adjective', 'is_adverb', 'is_function_word',
    'pos_dominance'
]

subtlex_correlations = {}
for feature in subtlex_features:
    corr, p_value = pearsonr(df_subtlex_merged[feature].fillna(0), df_subtlex_merged['complexity'])
    subtlex_correlations[feature] = corr

# Sort and display correlations
subtlex_corrs_sorted = sorted(subtlex_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nSUBTLEX feature correlations with complexity:")
for feature, corr in subtlex_corrs_sorted:
    print(f"{feature}: {corr:.4f}")

# Create visualization of correlations
plt.figure(figsize=(10, 6))
features, corrs = zip(*subtlex_corrs_sorted)
plt.barh(features, corrs)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('SUBTLEX Features: Correlation with Word Complexity')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =================================================================
# 2. MRC FEATURES (Linguistic Properties)
# =================================================================
print("\n" + "=" * 60)
print("GROUP 2: MRC FEATURES (89% overlap)")
print("=" * 60)

# Merge complexity with MRC
df_mrc_temp = df_mrc.copy()
df_mrc_temp['word_lower'] = df_mrc_temp['Word'].str.lower()
df_mrc_merged = pd.merge(df_complexity, df_mrc_temp, on='word_lower', how='left')
print(f"Merged shape: {df_mrc_merged.shape}")

# Create MRC-based features
# Rename for clarity
feature_mapping = {
    'Number of Letters': 'letter_count',
    'Number of Phonemes': 'phoneme_count',
    'Number of Syllables': 'syllable_count',
    'KF Written Frequency': 'written_frequency',
    'Familiarity': 'familiarity',
    'Concreteness': 'concreteness_mrc',
    'Imageability': 'imageability',
    'Age of Acquisition Rating': 'age_of_acquisition_mrc'
}

# Apply the mappings
for orig_col, new_col in feature_mapping.items():
    if orig_col in df_mrc_merged.columns:
        df_mrc_merged[new_col] = df_mrc_merged[orig_col]

# Calculate correlations with complexity
mrc_features = list(feature_mapping.values())
mrc_correlations = {}

for feature in mrc_features:
    if feature in df_mrc_merged.columns:  # Ensure the feature exists
        # Fill missing values with median for correlation calculation
        feature_data = df_mrc_merged[feature].fillna(df_mrc_merged[feature].median())
        corr, p_value = pearsonr(feature_data, df_mrc_merged['complexity'])
        mrc_correlations[feature] = corr

# Sort and display correlations
mrc_corrs_sorted = sorted(mrc_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nMRC feature correlations with complexity:")
for feature, corr in mrc_corrs_sorted:
    print(f"{feature}: {corr:.4f}")

# Create visualization of correlations
plt.figure(figsize=(10, 6))
features, corrs = zip(*mrc_corrs_sorted)
plt.barh(features, corrs)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('MRC Features: Correlation with Word Complexity')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =================================================================
# 3. AGE OF ACQUISITION FEATURES (AoA)
# =================================================================
print("\n" + "=" * 60)
print("GROUP 3: AGE OF ACQUISITION FEATURES (85% overlap)")
print("=" * 60)

# Merge complexity with AoA
df_aoa_temp = df_aoa.copy()
df_aoa_temp['word_lower'] = df_aoa_temp['Word'].str.lower()
df_aoa_merged = pd.merge(df_complexity, df_aoa_temp, on='word_lower', how='left')
print(f"Merged shape: {df_aoa_merged.shape}")

# Create AoA-based features
feature_mapping = {
    'AoA_Kup': 'age_of_acquisition',
    'Perc_known': 'percent_known',
    'Nletters': 'letter_count_aoa',
    'Nphon': 'phoneme_count_aoa',
    'Nsyll': 'syllable_count_aoa'
}

# Apply the mappings
for orig_col, new_col in feature_mapping.items():
    if orig_col in df_aoa_merged.columns:
        df_aoa_merged[new_col] = df_aoa_merged[orig_col]

# Calculate correlations with complexity
aoa_features = list(feature_mapping.values())
aoa_correlations = {}

for feature in aoa_features:
    if feature in df_aoa_merged.columns:  # Ensure the feature exists
        # Fill missing values with median for correlation calculation only
        feature_data = df_aoa_merged[feature].fillna(df_aoa_merged[feature].median())
        corr, p_value = pearsonr(feature_data, df_aoa_merged['complexity'])
        aoa_correlations[feature] = corr

# Sort and display correlations
aoa_corrs_sorted = sorted(aoa_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nAoA feature correlations with complexity:")
for feature, corr in aoa_corrs_sorted:
    print(f"{feature}: {corr:.4f}")

# Create visualization of correlations
plt.figure(figsize=(10, 6))
features, corrs = zip(*aoa_corrs_sorted)
plt.barh(features, corrs)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('Age of Acquisition Features: Correlation with Word Complexity')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =================================================================
# 4. CONCRETENESS FEATURES
# =================================================================
print("\n" + "=" * 60)
print("GROUP 4: CONCRETENESS FEATURES (56% overlap)")
print("=" * 60)

# Merge complexity with Concreteness
df_conc_temp = df_concreteness.copy()
df_conc_temp['word_lower'] = df_conc_temp['Word'].str.lower()
df_conc_merged = pd.merge(df_complexity, df_conc_temp, on='word_lower', how='left')
print(f"Merged shape: {df_conc_merged.shape}")

# Create Concreteness-based features
feature_mapping = {
    'Conc.M': 'concreteness_rating',
    'Conc.SD': 'concreteness_sd',
    'Percent_known': 'percent_known_conc'
}

# Apply the mappings
for orig_col, new_col in feature_mapping.items():
    if orig_col in df_conc_merged.columns:
        df_conc_merged[new_col] = df_conc_merged[orig_col]

# Calculate correlations with complexity
conc_features = list(feature_mapping.values())
conc_correlations = {}

for feature in conc_features:
    if feature in df_conc_merged.columns:  # Ensure the feature exists
        # Fill missing values with median for correlation calculation only
        feature_data = df_conc_merged[feature].fillna(df_conc_merged[feature].median())
        corr, p_value = pearsonr(feature_data, df_conc_merged['complexity'])
        conc_correlations[feature] = corr

# Sort and display correlations
conc_corrs_sorted = sorted(conc_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nConcreteness feature correlations with complexity:")
for feature, corr in conc_corrs_sorted:
    print(f"{feature}: {corr:.4f}")

# Create visualization of correlations
plt.figure(figsize=(10, 6))
features, corrs = zip(*conc_corrs_sorted)
plt.barh(features, corrs)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('Concreteness Features: Correlation with Word Complexity')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =================================================================
# 5. CROSS-DATABASE COMBINED FEATURES
# =================================================================
print("\n" + "=" * 60)
print("GROUP 5: CROSS-DATABASE COMBINED FEATURES")
print("=" * 60)

# Identify top features from each database
top_subtlex = [feature for feature, corr in subtlex_corrs_sorted[:3]]
top_mrc = [feature for feature, corr in mrc_corrs_sorted[:3]]
top_aoa = [feature for feature, corr in aoa_corrs_sorted[:3]]
top_conc = [feature for feature, corr in conc_corrs_sorted[:3]]

print("\nTop features from each database:")
print(f"SUBTLEX: {top_subtlex}")
print(f"MRC: {top_mrc}")
print(f"AoA: {top_aoa}")
print(f"Concreteness: {top_conc}")

# Create a final feature dataframe with top features
print("\nCreating final combined feature set...")

# Start with the complexity dataset
final_features = df_complexity.copy()

# Add top SUBTLEX features
for feature in top_subtlex:
    final_features[feature] = df_subtlex_merged[feature]

# Add top MRC features
for feature in top_mrc:
    if feature in df_mrc_merged.columns:
        final_features[feature] = df_mrc_merged[feature]

# Add top AoA features
for feature in top_aoa:
    if feature in df_aoa_merged.columns:
        final_features[feature] = df_aoa_merged[feature]

# Add top Concreteness features
for feature in top_conc:
    if feature in df_conc_merged.columns:
        final_features[feature] = df_conc_merged[feature]

# Explore missing values in the combined dataset
missing_counts = final_features.isnull().sum()
print("\nMissing values in final feature set:")
for col, count in missing_counts.items():
    if count > 0:
        percent = count / len(final_features) * 100
        print(f"{col}: {count} missing values ({percent:.2f}%)")

# Save the final feature set WITHOUT imputation
# This allows Cell 8 to perform more sophisticated imputation
final_features.to_csv('data/word_complexity_combined_features.csv', index=False)
print(f"\nFinal feature set shape: {final_features.shape}")
print("Saved to: data/word_complexity_combined_features.csv")

# Create correlation matrix heatmap for the top features
print("\nCreating correlation matrix for top features...")
top_features = top_subtlex + top_mrc + top_aoa + top_conc
# Get only those that actually exist in our dataframe
top_features = [f for f in top_features if f in final_features.columns]

# Calculate correlation matrix
# For correlation calculation only, temporarily fill missing values
corr_df = final_features[top_features + ['complexity']].copy()
for col in corr_df.columns:
    if col != 'complexity' and corr_df[col].isnull().sum() > 0:
        corr_df[col] = corr_df[col].fillna(corr_df[col].median())
        
corr_matrix = corr_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Top Psycholinguistic Features')
plt.tight_layout()
plt.show()

# Print summary of findings
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)
print("The most predictive psycholinguistic features for word complexity are:")

# Combine all correlations
all_correlations = {**subtlex_correlations, **mrc_correlations, 
                    **aoa_correlations, **conc_correlations}

# Sort all correlations by absolute value
all_corrs_sorted = sorted(all_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# Print top 10 overall features
print("\nTop 10 features across all databases:")
for i, (feature, corr) in enumerate(all_corrs_sorted[:10], 1):
    print(f"{i}. {feature}: {corr:.4f}")


# In[22]:

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------

# Cell 7: Create and Evaluate Advanced Word Complexity Features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

print("Creating and evaluating advanced word complexity features...")

# Load the merged feature set we created previously
df_features = pd.read_csv('data/word_complexity_combined_features.csv')
print(f"Loaded feature dataset with shape: {df_features.shape}")

# Check that all key features are available
required_features = ['complexity', 'cd_log10', 'freq_log10', 'zipf_value', 
                     'age_of_acquisition', 'percent_known', 'familiarity', 
                     'imageability', 'concreteness_mrc', 'syllable_count_aoa']

missing_features = [f for f in required_features if f not in df_features.columns]
if missing_features:
    print(f"Warning: The following required features are missing: {missing_features}")
    # If we're missing some key features, we'll need to recreate them using the original databases
    # For this code snippet, we'll assume all required features are present

# 1. Create transformed features
print("\nCreating transformed features...")
# Log transform for age of acquisition (add small constant to avoid log(0))
df_features['log_age_of_acquisition'] = np.log1p(df_features['age_of_acquisition'])

# Inverse frequency 
df_features['inverse_frequency'] = 1 / (df_features['freq_log10'] + 1)  # Add 1 to avoid division by zero

# Square root of syllable count
if 'syllable_count' in df_features.columns:
    df_features['sqrt_syllable_count'] = np.sqrt(df_features['syllable_count'])
elif 'syllable_count_aoa' in df_features.columns:
    df_features['sqrt_syllable_count'] = np.sqrt(df_features['syllable_count_aoa'])

# 2. Create interaction features
print("Creating interaction features...")
# Frequency × Age of Acquisition
df_features['freq_by_aoa'] = df_features['freq_log10'] * df_features['age_of_acquisition']

# Imageability × Frequency
df_features['imageability_by_frequency'] = df_features['imageability'] * df_features['freq_log10']

# Concreteness × Age of Acquisition
df_features['concreteness_by_aoa'] = df_features['concreteness_mrc'] * df_features['age_of_acquisition']

# 3. Create ratio features
print("Creating ratio features...")
# Familiarity to frequency ratio
df_features['familiarity_to_frequency_ratio'] = df_features['familiarity'] / (df_features['freq_log10'] + 1)  # Add 1 to avoid division by zero

# Syllable to letter ratio
if 'letter_count' in df_features.columns and 'syllable_count' in df_features.columns:
    df_features['syllable_to_letter_ratio'] = df_features['syllable_count'] / df_features['letter_count']
elif 'letter_count_aoa' in df_features.columns and 'syllable_count_aoa' in df_features.columns:
    df_features['syllable_to_letter_ratio'] = df_features['syllable_count_aoa'] / df_features['letter_count_aoa']

# 4. Create combined complexity indices
print("Creating combined complexity indices...")
# Cognitive load index
df_features['cognitive_load_index'] = (
    df_features['age_of_acquisition'] + 
    (10 - df_features['freq_log10']) +  # Invert frequency (higher = more complex)
    df_features['syllable_count_aoa']
)

# Conceptual difficulty
df_features['conceptual_difficulty'] = (
    (600 - df_features['imageability']) / 100 +  # Rescale and invert (higher = more abstract)
    (600 - df_features['concreteness_mrc']) / 100 +  # Rescale and invert (higher = more abstract)
    df_features['age_of_acquisition'] / 10  # Scale to similar range
)

# Calculate correlations for all features with complexity
print("\nCalculating correlations with complexity...")
all_features = [col for col in df_features.columns 
                if col not in ['id', 'corpus', 'sentence', 'token', 'word_lower', 'complexity']]

correlations = {}
for feature in all_features:
    # Skip non-numeric columns
    if df_features[feature].dtype not in ['float64', 'int64']:
        continue
    
    # Fill NaN values with median for correlation calculation
    data = df_features[feature].fillna(df_features[feature].median())
    try:
        corr, p_value = pearsonr(data, df_features['complexity'])
        corr_spear, p_value_spear = spearmanr(data, df_features['complexity'])
        correlations[feature] = (corr, p_value, corr_spear, p_value_spear)
    except:
        print(f"Error calculating correlation for feature: {feature}")

# Sort features by absolute Pearson correlation
sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

# Print all correlations, grouped by feature type
print("\n" + "=" * 80)
print("CORRELATIONS WITH WORD COMPLEXITY")
print("=" * 80)

# Group features by type
feature_groups = {
    "Base Features": ['cd_log10', 'freq_log10', 'zipf_value', 'age_of_acquisition', 
                      'percent_known', 'familiarity', 'imageability', 'concreteness_mrc',
                      'syllable_count_aoa', 'letter_count_aoa', 'phoneme_count_aoa'],
    "Transformed Features": ['log_age_of_acquisition', 'inverse_frequency', 'sqrt_syllable_count'],
    "Interaction Features": ['freq_by_aoa', 'imageability_by_frequency', 'concreteness_by_aoa'],
    "Ratio Features": ['familiarity_to_frequency_ratio', 'syllable_to_letter_ratio'],
    "Combined Indices": ['cognitive_load_index', 'conceptual_difficulty']
}

for group, features in feature_groups.items():
    print(f"\n{group}:")
    print(f"{'Feature':<30} {'Pearson':<10} {'p-value':<10} {'Spearman':<10} {'p-value':<10}")
    print("-" * 70)
    
    for feature in features:
        if feature in correlations:
            corr, p, corr_s, p_s = correlations[feature]
            print(f"{feature:<30} {corr:>+.4f}    {p:<10.4e} {corr_s:>+.4f}    {p_s:<10.4e}")
        else:
            print(f"{feature:<30} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

# Create visualization of all correlations
plt.figure(figsize=(14, 10))
features = [x[0] for x in sorted_features]
corrs = [x[1][0] for x in sorted_features]  # Pearson correlations

# Create barplot with color coding by feature type
colors = []
for feature in features:
    if feature in feature_groups["Base Features"]:
        colors.append('blue')
    elif feature in feature_groups["Transformed Features"]:
        colors.append('green')
    elif feature in feature_groups["Interaction Features"]:
        colors.append('orange')
    elif feature in feature_groups["Ratio Features"]:
        colors.append('red')
    elif feature in feature_groups["Combined Indices"]:
        colors.append('purple')
    else:
        colors.append('gray')

# Create barplot
plt.barh(range(len(features)), corrs, color=colors)
plt.yticks(range(len(features)), features)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('Feature Correlations with Word Complexity')
plt.xlabel('Pearson Correlation')

# Create legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', label='Base Features'),
    Patch(facecolor='green', label='Transformed Features'),
    Patch(facecolor='orange', label='Interaction Features'),
    Patch(facecolor='red', label='Ratio Features'),
    Patch(facecolor='purple', label='Combined Indices')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# Identify top features overall
print("\n" + "=" * 80)
print("TOP 15 FEATURES FOR WORD COMPLEXITY PREDICTION")
print("=" * 80)
print(f"{'Feature':<30} {'Pearson':<10} {'p-value':<10} {'Spearman':<10} {'p-value':<10}")
print("-" * 70)

for feature, (corr, p, corr_s, p_s) in sorted_features[:15]:
    print(f"{feature:<30} {corr:>+.4f}    {p:<10.4e} {corr_s:>+.4f}    {p_s:<10.4e}")

# Identify top features from each group
print("\n" + "=" * 80)
print("TOP FEATURE FROM EACH GROUP")
print("=" * 80)

for group, features in feature_groups.items():
    # Find the feature with the highest absolute correlation in this group
    max_corr = 0
    top_feature = None
    
    for feature in features:
        if feature in correlations:
            corr = abs(correlations[feature][0])
            if corr > max_corr:
                max_corr = corr
                top_feature = feature
    
    if top_feature:
        corr, p, corr_s, p_s = correlations[top_feature]
        print(f"{group + ':':<20} {top_feature:<30} {corr:>+.4f}")

# Final feature recommendations
print("\n" + "=" * 80)
print("RECOMMENDED FEATURES FOR WORD COMPLEXITY PREDICTION")
print("=" * 80)

# Get top 2 features from each group
recommended_features = []
for group, features in feature_groups.items():
    group_corrs = [(f, abs(correlations[f][0])) for f in features if f in correlations]
    if group_corrs:
        group_corrs.sort(key=lambda x: x[1], reverse=True)
        recommended_features.extend([f[0] for f in group_corrs[:min(2, len(group_corrs))]])

# Print final recommendations with correlations
print("Based on correlation analysis, these are the recommended features for the final model:")
for feature in recommended_features:
    corr, p, corr_s, p_s = correlations[feature]
    print(f"- {feature:<30} {corr:>+.4f}")

# Save enhanced feature set
df_features.to_csv('data/word_complexity_enhanced_features.csv', index=False)
print("\nEnhanced feature set saved to data/word_complexity_enhanced_features.csv")


# In[32]:
# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load the enhanced feature dataset
df_features = pd.read_csv('data/word_complexity_enhanced_features.csv')
print(f"Loaded dataset with {df_features.shape[0]} words and {df_features.shape[1]} features")

# Identify ALL features that might need imputation (including base features)
print("\nMissing values before imputation:")
missing_counts = df_features.isnull().sum()
for col, count in missing_counts.items():
    if count > 0:
        percent = 100 * count / len(df_features)
        print(f"{col}: {count} missing values ({percent:.2f}%)")

# Define more comprehensive lists of features requiring imputation
# 1. Base features (primary features that aren't derived from others)
base_features_to_impute = [
    'cd_log10', 'freq_log10', 'zipf_value',  # Frequency features
    'familiarity', 'imageability', 'concreteness_mrc',  # MRC features
    'age_of_acquisition', 'percent_known', 'syllable_count_aoa',  # AoA features
    'percent_known_conc', 'concreteness_sd', 'concreteness_rating'  # Concreteness features
]

# 2. Derived features (calculated from other features)
derived_features_to_impute = [
    'log_age_of_acquisition',
    'inverse_frequency',
    'sqrt_syllable_count',
    'freq_by_aoa',
    'imageability_by_frequency',
    'concreteness_by_aoa',
    'familiarity_to_frequency_ratio',
    'cognitive_load_index',
    'conceptual_difficulty'
]

# Define specialized imputation methods for each feature type
def impute_frequency_features(df, feature):
    """
    Imputation method for frequency-related features.
    
    Reasoning: Word frequency features are highly correlated with each other,
    and also correlate with word length and syllable count.
    """
    print(f"\nImputing frequency feature: {feature}...")
    
    # Identify predictors for frequency
    potential_predictors = ['word_length', 'num_syllables', 'syllable_count_aoa']
    available_predictors = [p for p in potential_predictors if p in df.columns and df[p].isnull().sum() == 0]
    
    # If we have other frequency features available, use those primarily
    other_freq_features = [f for f in ['cd_log10', 'freq_log10', 'zipf_value'] 
                           if f in df.columns and f != feature and df[f].isnull().sum() == 0]
    
    predictors = other_freq_features + available_predictors
    
    if len(predictors) < 1:
        print(f"  Not enough complete predictors for {feature}. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    # Create training data from words that have the feature
    train_mask = df[feature].notna()
    
    # Fit a linear regression model on words with known values
    X_train = df.loc[train_mask, predictors]
    y_train = df.loc[train_mask, feature]
    
    # Handle case where all predictors are constant
    if X_train.nunique().min() <= 1:
        print(f"  Predictor has constant values. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict missing values
    pred_mask = df[feature].isna()
    if pred_mask.sum() > 0:
        X_pred = df.loc[pred_mask, predictors]
        df.loc[pred_mask, feature] = model.predict(X_pred)
        
        # Print model coefficients for transparency
        coef_dict = {pred: coef for pred, coef in zip(predictors, model.coef_)}
        print(f"  Regression coefficients: {coef_dict}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  Imputed {pred_mask.sum()} values")
    
    return df[feature]

def impute_mrc_features(df, feature):
    """
    Imputation method for MRC psycholinguistic database features.
    
    Reasoning: MRC features like familiarity and imageability show patterns based on
    word frequency, word length, and part of speech.
    """
    print(f"\nImputing MRC feature: {feature}...")
    
    # Identify predictors for MRC features
    potential_predictors = [
        'cd_log10', 'freq_log10', 'word_length', 'num_syllables', 
        'is_noun', 'is_verb', 'is_adjective', 'is_adverb'
    ]
    
    # Add other MRC features that might be useful predictors
    other_mrc_features = [f for f in ['familiarity', 'imageability', 'concreteness_mrc'] 
                         if f in df.columns and f != feature and df[f].isnull().sum() == 0]
    
    available_predictors = [p for p in potential_predictors 
                           if p in df.columns and df[p].isnull().sum() == 0]
    predictors = available_predictors + other_mrc_features
    
    if len(predictors) < 2:
        print(f"  Not enough complete predictors for {feature}. Using KNN imputation with basic features.")
        # Use KNN with basic features that should always be available
        basic_predictors = ['word_length', 'num_syllables']
        basic_predictors = [p for p in basic_predictors if p in df.columns]
        
        # Prepare data for KNN
        impute_cols = basic_predictors + [feature]
        impute_df = df[impute_cols].copy()
        
        # Scale the data
        scaler = StandardScaler()
        impute_df[basic_predictors] = scaler.fit_transform(impute_df[basic_predictors])
        
        # KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(impute_df)
        
        # Update only the missing values
        missing_mask = df[feature].isna()
        if missing_mask.sum() > 0:
            result = df[feature].copy()
            result[missing_mask] = pd.DataFrame(
                imputed, columns=impute_cols
            )[feature].iloc[missing_mask.values].values
            
            print(f"  Imputed {missing_mask.sum()} values using KNN with basic predictors")
            return result
        
        return df[feature]
    
    # If we have enough predictors, use regression
    train_mask = df[feature].notna()
    X_train = df.loc[train_mask, predictors]
    y_train = df.loc[train_mask, feature]
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict missing values
    pred_mask = df[feature].isna()
    if pred_mask.sum() > 0:
        X_pred = df.loc[pred_mask, predictors]
        df.loc[pred_mask, feature] = model.predict(X_pred)
        
        # Print model coefficients for transparency
        coef_dict = {pred: coef for pred, coef in zip(predictors, model.coef_)}
        print(f"  Regression coefficients: {coef_dict}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  Imputed {pred_mask.sum()} values")
    
    return df[feature]

def impute_age_features(df, feature):
    """
    Imputation method for age of acquisition features.
    
    Reasoning: Age of acquisition strongly correlates with word frequency and word length.
    We use regression-based imputation to estimate AoA values based on these correlations.
    """
    print(f"\nImputing age feature: {feature}...")
    
    # Create training data from words that have the feature
    train_mask = df[feature].notna()
    
    # For prediction, we use frequency and word length as predictors
    potential_predictors = ['freq_log10', 'cd_log10', 'word_length', 'num_syllables', 'syllable_count_aoa']
    available_predictors = [p for p in potential_predictors 
                           if p in df.columns and df[p].isnull().sum() == 0]
    
    if len(available_predictors) < 1:
        print(f"  Not enough complete predictors for {feature}. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    # Fit a linear regression model on words with known values
    X_train = df.loc[train_mask, available_predictors]
    y_train = df.loc[train_mask, feature]
    
    # Handle case where all predictors are constant
    if X_train.nunique().min() <= 1:
        print(f"  Predictor has constant values. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict missing values
    pred_mask = df[feature].isna()
    if pred_mask.sum() > 0:
        X_pred = df.loc[pred_mask, available_predictors]
        df.loc[pred_mask, feature] = model.predict(X_pred)
        
        # Print model coefficients for transparency
        coef_dict = {pred: coef for pred, coef in zip(available_predictors, model.coef_)}
        print(f"  Regression coefficients: {coef_dict}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  Imputed {pred_mask.sum()} values")
    
    return df[feature]

def impute_semantic_features(df, feature):
    """
    Imputation method for semantic features from WordNet.
    
    Reasoning: Semantic features like hypernyms and hyponyms relate to word 
    conceptual structure. They correlate with concreteness, imageability,
    and part of speech. We use KNN imputation to find semantically similar words.
    """
    print(f"\nImputing semantic feature: {feature}...")
    
    # Identify features useful for finding semantically similar words
    semantic_predictors = [
        'freq_log10', 'syllable_count_aoa', 'concreteness_mrc', 
        'imageability', 'is_noun', 'is_verb', 'is_adjective'
    ]
    available_predictors = [p for p in semantic_predictors 
                           if p in df.columns and df[p].isnull().sum() == 0]
    
    if len(available_predictors) < 2:
        print(f"  Not enough predictors for KNN. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    # Prepare data for KNN
    impute_cols = available_predictors + [feature]
    impute_df = df[impute_cols].copy()
    
    # Scale the data
    scaler = StandardScaler()
    impute_df[available_predictors] = scaler.fit_transform(impute_df[available_predictors].fillna(0))
    
    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(impute_df)
    
    # Update only the missing values
    missing_mask = df[feature].isna()
    if missing_mask.sum() > 0:
        result = df[feature].copy()
        result[missing_mask] = pd.DataFrame(
            imputed, columns=impute_cols
        )[feature].iloc[missing_mask.values].values
        
        print(f"  Imputed {missing_mask.sum()} values using KNN with predictors: {available_predictors}")
        return result
    
    return df[feature]

def impute_concreteness_features(df, feature):
    """
    Imputation method for concreteness-related features.
    
    Reasoning: Concreteness is related to imageability, familiarity,
    and part of speech (e.g., nouns tend to be more concrete than verbs).
    """
    print(f"\nImputing concreteness feature: {feature}...")
    
    # Identify predictors for concreteness features
    potential_predictors = [
        'imageability', 'familiarity', 'age_of_acquisition',
        'is_noun', 'is_verb', 'is_adjective', 'is_adverb',
        'freq_log10', 'word_length'
    ]
    
    # Add other concreteness features that might be useful predictors
    other_conc_features = [f for f in ['concreteness_rating', 'concreteness_sd', 'percent_known_conc'] 
                         if f in df.columns and f != feature and df[f].isnull().sum() == 0]
    
    available_predictors = [p for p in potential_predictors 
                           if p in df.columns and df[p].isnull().sum() == 0]
    predictors = available_predictors + other_conc_features
    
    if len(predictors) < 2:
        print(f"  Not enough complete predictors for {feature}. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    # Create training data from words that have the feature
    train_mask = df[feature].notna()
    
    # Fit a linear regression model on words with known values
    X_train = df.loc[train_mask, predictors]
    y_train = df.loc[train_mask, feature]
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict missing values
    pred_mask = df[feature].isna()
    if pred_mask.sum() > 0:
        X_pred = df.loc[pred_mask, predictors]
        df.loc[pred_mask, feature] = model.predict(X_pred)
        
        # Print model coefficients for transparency
        coef_dict = {pred: coef for pred, coef in zip(predictors, model.coef_)}
        print(f"  Regression coefficients: {coef_dict}")
        print(f"  Intercept: {model.intercept_:.4f}")
        print(f"  Imputed {pred_mask.sum()} values")
    
    return df[feature]

def impute_derived_features(df, feature, component_features):
    """
    Imputation method for derived features (those calculated from other features).
    
    Reasoning: For derived features, the best approach is to recalculate them
    after imputing their component features. This maintains the mathematical
    relationship between features.
    """
    print(f"\nImputing derived feature: {feature} by recalculating from its components...")
    
    # Check if all component features are available
    missing_components = [f for f in component_features if f not in df.columns]
    if missing_components:
        print(f"  Cannot recalculate {feature}. Missing components: {missing_components}")
        # Fall back to median imputation
        return df[feature].fillna(df[feature].median())
    
    # Check if any component features still have missing values
    components_with_nulls = []
    for comp in component_features:
        null_count = df[comp].isnull().sum()
        if null_count > 0:
            components_with_nulls.append((comp, null_count))
    
    if components_with_nulls:
        print(f"  Warning: Some component features still have missing values:")
        for comp, count in components_with_nulls:
            print(f"    - {comp}: {count} missing values")
        print("  Will attempt calculation but result will have missing values where components are missing")
    
    # Determine the formula based on the feature name
    if feature == 'log_age_of_acquisition':
        # Log transform with adjustment for zeros
        result = np.log1p(df['age_of_acquisition'])
        print("  Recalculated as: log(1 + age_of_acquisition)")
    
    elif feature == 'inverse_frequency':
        # Inverse frequency with adjustment to avoid division by zero
        result = 1 / (df['freq_log10'] + 1)
        print("  Recalculated as: 1 / (freq_log10 + 1)")
    
    elif feature == 'sqrt_syllable_count':
        if 'syllable_count' in df.columns:
            result = np.sqrt(df['syllable_count'])
            print("  Recalculated as: sqrt(syllable_count)")
        elif 'syllable_count_aoa' in df.columns:
            result = np.sqrt(df['syllable_count_aoa'])
            print("  Recalculated as: sqrt(syllable_count_aoa)")
        else:
            print("  No syllable count column found. Using median imputation.")
            return df[feature].fillna(df[feature].median())
    
    elif feature == 'freq_by_aoa':
        result = df['freq_log10'] * df['age_of_acquisition']
        print("  Recalculated as: freq_log10 * age_of_acquisition")
    
    elif feature == 'imageability_by_frequency':
        result = df['imageability'] * df['freq_log10']
        print("  Recalculated as: imageability * freq_log10")
    
    elif feature == 'concreteness_by_aoa':
        result = df['concreteness_mrc'] * df['age_of_acquisition']
        print("  Recalculated as: concreteness_mrc * age_of_acquisition")
    
    elif feature == 'familiarity_to_frequency_ratio':
        # Avoid division by zero by adding a small constant
        result = df['familiarity'] / (df['freq_log10'] + 1)
        print("  Recalculated as: familiarity / (freq_log10 + 1)")
    
    elif feature == 'cognitive_load_index':
        result = (df['age_of_acquisition'] + (10 - df['freq_log10']) + df['syllable_count_aoa'])
        print("  Recalculated as: age_of_acquisition + (10 - freq_log10) + syllable_count_aoa")
    
    elif feature == 'conceptual_difficulty':
        result = ((600 - df['imageability']) / 100 + 
                  (600 - df['concreteness_mrc']) / 100 + 
                  df['age_of_acquisition'] / 10)
        print("  Recalculated as: (600-imageability)/100 + (600-concreteness_mrc)/100 + age_of_acquisition/10")
    
    else:
        print(f"  No recalculation formula defined for {feature}. Using median imputation.")
        return df[feature].fillna(df[feature].median())
    
    # Only update missing values
    missing_mask = df[feature].isna()
    if missing_mask.sum() > 0:
        final_result = df[feature].copy()
        final_result[missing_mask] = result[missing_mask]
        print(f"  Imputed {missing_mask.sum()} values by recalculation")
        
        # Check if the recalculation introduced any new missing values
        new_missing = final_result.isna().sum() - df[feature].isna().sum()
        if new_missing > 0:
            print(f"  Warning: Recalculation introduced {new_missing} new missing values")
            print("  These are likely due to missing values in component features")
            print("  Will fill remaining missing values with median")
            final_result = final_result.fillna(df[feature].dropna().median())
            print(f"  Filled {new_missing} values with median: {df[feature].dropna().median()}")
            
        return final_result
    
    return df[feature]

# Step 1: Impute base features first
print("\nStep 1: Imputing base features...")

# 1a. Impute frequency features
for feature in ['cd_log10', 'freq_log10', 'zipf_value']:
    if feature in df_features.columns and df_features[feature].isnull().sum() > 0:
        df_features[feature] = impute_frequency_features(df_features, feature)

# 1b. Impute age of acquisition features
for feature in ['age_of_acquisition', 'percent_known', 'syllable_count_aoa']:
    if feature in df_features.columns and df_features[feature].isnull().sum() > 0:
        df_features[feature] = impute_age_features(df_features, feature)

# 1c. Impute MRC features
for feature in ['familiarity', 'imageability', 'concreteness_mrc']:
    if feature in df_features.columns and df_features[feature].isnull().sum() > 0:
        df_features[feature] = impute_mrc_features(df_features, feature)

# 1d. Impute concreteness features
for feature in ['percent_known_conc', 'concreteness_sd', 'concreteness_rating']:
    if feature in df_features.columns and df_features[feature].isnull().sum() > 0:
        df_features[feature] = impute_concreteness_features(df_features, feature)

# Check if base features have been properly imputed
print("\nChecking base features after imputation:")
base_features_null_counts = {f: df_features[f].isnull().sum() 
                            for f in base_features_to_impute 
                            if f in df_features.columns}
for feature, null_count in base_features_null_counts.items():
    if null_count > 0:
        print(f"Warning: {feature} still has {null_count} missing values!")
        print(f"Will attempt to impute these with median before proceeding")
        df_features[feature] = df_features[feature].fillna(df_features[feature].dropna().median())
        print(f"Filled with median: {df_features[feature].dropna().median()}")

# Step 2: Impute derived features using the now-complete base features
print("\nStep 2: Imputing derived features...")

# Define component dependencies for each derived feature
derived_feature_components = {
    'log_age_of_acquisition': ['age_of_acquisition'],
    'inverse_frequency': ['freq_log10'],
    'sqrt_syllable_count': ['syllable_count_aoa'],
    'freq_by_aoa': ['freq_log10', 'age_of_acquisition'],
    'imageability_by_frequency': ['imageability', 'freq_log10'],
    'concreteness_by_aoa': ['concreteness_mrc', 'age_of_acquisition'],
    'familiarity_to_frequency_ratio': ['familiarity', 'freq_log10'],
    'cognitive_load_index': ['age_of_acquisition', 'freq_log10', 'syllable_count_aoa'],
    'conceptual_difficulty': ['imageability', 'concreteness_mrc', 'age_of_acquisition']
}

# Impute each derived feature using its components
for feature, components in derived_feature_components.items():
    if feature in df_features.columns and df_features[feature].isnull().sum() > 0:
        df_features[feature] = impute_derived_features(df_features, feature, components)

# Check if any features still have missing values
print("\nMissing values after imputation:")
for feature in base_features_to_impute + derived_features_to_impute:
    if feature in df_features.columns:
        missing = df_features[feature].isnull().sum()
        percent = 100 * missing / len(df_features)
        print(f"{feature}: {missing} missing values ({percent:.2f}%)")

# Final check for any remaining missing values
missing_counts_after = df_features.isnull().sum()
features_still_missing = {col: count for col, count in missing_counts_after.items() if count > 0}

if features_still_missing:
    print("\nWarning: Some features still have missing values after imputation!")
    for feature, count in features_still_missing.items():
        print(f"{feature}: {count} missing values")
    
    print("\nPerforming final median imputation for any remaining missing values...")
    for feature in features_still_missing:
        if feature not in ['id', 'corpus', 'sentence', 'token', 'complexity', 'word_lower']:
            df_features[feature] = df_features[feature].fillna(df_features[feature].dropna().median())
    
    # Verify all missing values are filled
    final_missing = df_features.isnull().sum()
    if final_missing.sum() > 0:
        print("There are still missing values after final imputation:")
        for col, count in final_missing.items():
            if count > 0:
                print(f"{col}: {count} missing values")
    else:
        print("All missing values have been successfully imputed!")


# In[34]:

# ------------------------------------------------------------------------------------------ BREAK ------------------------------------------------------------------------------------------

# Cell 9: Word Complexity Prediction Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import xgboost as XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')

print("Building and evaluating word complexity prediction models...")

# =================================================================
# 1. DATA PREPARATION
# =================================================================
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Define target and features
target_col = 'complexity'
y = df_features[target_col]

# Define feature groups for comparison
feature_groups = {
    # Group 1: Traditional linguistic features
    "traditional_features": [
        'word_length', 'num_syllables', 'num_vowels', 'num_consonants', 
        'syllables_per_char', 'is_capitalized', 'is_all_caps'
    ],
    
    # Group 2: Frequency features
    "frequency_features": [
        'cd_log10', 'freq_log10', 'zipf_value', 'inverse_frequency'
    ],
    
    # Group 3: Psycholinguistic features
    "psycholinguistic_features": [
        'age_of_acquisition', 'familiarity', 'imageability', 'concreteness_mrc',
        'log_age_of_acquisition', 'percent_known'
    ],
    
    # Group 4: Combined indices
    "combined_indices": [
        'cognitive_load_index', 'conceptual_difficulty', 'freq_by_aoa',
        'imageability_by_frequency', 'familiarity_to_frequency_ratio'
    ],
    
    # Group 5: Corpus indicators
    "corpus_features": [
        'is_biomed', 'is_europarl', 'is_bible'
    ]
}

# Create combined feature sets
feature_groups["frequency_and_psycholinguistic"] = (
    feature_groups["frequency_features"] + 
    feature_groups["psycholinguistic_features"]
)

feature_groups["all_features"] = []
for group in feature_groups.values():
    feature_groups["all_features"].extend(group)
feature_groups["all_features"] = list(set(feature_groups["all_features"]))  # remove duplicates

# Filter out features not in the dataframe
for group_name, features in feature_groups.items():
    feature_groups[group_name] = [f for f in features if f in df_features.columns]
    print(f"Features in {group_name}: {len(feature_groups[group_name])}")

# Split data into train, validation, and test sets (60%/20%/20%)
X = df_features[feature_groups["all_features"]]

# Use stratified split based on corpus to ensure each corpus is represented
stratify_col = 'corpus'
if stratify_col in df_features.columns:
    print("\nUsing stratified split based on corpus...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df_features[stratify_col]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, 
        stratify=df_features.loc[X_temp.index, stratify_col]
    )
else:
    print("\nUsing regular random split...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nData split complete:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")


# In[35]:


# =================================================================
# 2. BASELINE MODEL EVALUATION
# =================================================================
print("\n" + "=" * 60)
print("BASELINE MODEL EVALUATION")
print("=" * 60)

# Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
}

try:
    import xgboost as xgb
    models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    print("XGBoost is available and will be included in the models.")
except ImportError:
    print("XGBoost is not available. Skipping this model.")

# Create a results table
results = []

# For each feature group, train and evaluate all models
for group_name, features in feature_groups.items():
    print(f"\nEvaluating models using {group_name}...")
    
    # Check if feature set is empty
    if not features:
        print(f"  Skipping {group_name} - no features found in this group")
        continue
    
    # Select feature set
    X_train_group = X_train[features]
    X_val_group = X_val[features]
    
    # Check if feature set is empty
    if X_train_group.empty or X_train_group.shape[1] == 0:
        print(f"  Skipping {group_name} - no valid features found in this group")
        continue
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_group)
    X_val_scaled = scaler.transform(X_val_group)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_val_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)
            
            # Calculate training time
            train_time = time.time() - start_time
            
            # Add results to table
            results.append({
                'Feature Group': group_name,
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Training Time (s)': train_time
            })
            
            # Print progress
            print(f"  {model_name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
        except Exception as e:
            print(f"  Error with {model_name} on {group_name}: {str(e)}")

# Check if we have any results
if not results:
    print("\nNo models could be successfully trained. Please check your data and feature groups.")
else:
    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['RMSE'])

    # Print best results overall
    print("\nTop 5 models across all feature groups based on validation RMSE:")
    if len(results_df) >= 5:
        print(results_df.head(5).to_string(index=False))
    else:
        print(results_df.to_string(index=False))

    # Print best results by feature group
    print("\nBest model for each feature group:")
    for group in feature_groups.keys():
        group_df = results_df[results_df['Feature Group'] == group].sort_values('RMSE')
        if not group_df.empty:
            print(f"\n{group}:")
            print(group_df.head(1).to_string(index=False))

    # Visualize model performance across feature groups
    if len(results_df) > 0:
        try:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='Feature Group', y='RMSE', data=results_df)
            plt.title('Model Performance by Feature Group')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            # Visualize best model performance
            top_models = results_df.groupby(['Feature Group', 'Model']).agg({'RMSE': 'min'}).reset_index()
            top_models = top_models.sort_values('RMSE')[:min(10, len(top_models))]
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='RMSE', y='Model', hue='Feature Group', data=top_models)
            plt.title('Top Model Configurations')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


# In[37]:


# =================================================================
# 3. MODEL TUNING FOR BEST PERFORMERS
# =================================================================
print("\n" + "=" * 60)
print("MODEL TUNING FOR BEST PERFORMERS")
print("=" * 60)

# Identify the best model type and feature group from baseline
best_result = results_df.iloc[0]
best_model_name = best_result['Model']
best_feature_group = best_result['Feature Group']

print(f"Tuning the best model: {best_model_name} with {best_feature_group}")

# Select features for best model
best_features = feature_groups[best_feature_group]
X_train_best = X_train[best_features]
X_val_best = X_val[best_features]
X_test_best = X_test[best_features]

# Define hyperparameter grids for each model type
param_grids = {
    "Linear Regression": {},  # No hyperparameters to tune
    
    "Ridge Regression": {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    },
    
    "Lasso Regression": {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    },
    
    "ElasticNet": {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6]
    },
    
    "SVR": {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    },
    
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5]
    }
}

# Create pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', models[best_model_name])
])

# Set up and perform grid search
param_grid = {'model__' + key: value for key, value in param_grids[best_model_name].items()}
if not param_grid:  # If empty (like for Linear Regression)
    print("No hyperparameters to tune for this model.")
    tuned_model = pipeline
else:
    print(f"Searching for best hyperparameters...")
    print(f"Parameter grid: {param_grid}")
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    
    grid_search.fit(X_train_best, y_train)
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best CV score (neg_rmse): {grid_search.best_score_:.4f}")
    
    tuned_model = grid_search.best_estimator_

# Evaluate tuned model on validation set
y_val_pred = tuned_model.predict(X_val_best)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"\nTuned model performance on validation set:")
print(f"RMSE: {rmse_val:.4f}")
print(f"MAE: {mae_val:.4f}")
print(f"R²: {r2_val:.4f}")

# Feature importance for appropriate models
if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    model = tuned_model.named_steps['model']
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': best_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
elif best_model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"]:
    model = tuned_model.named_steps['model']
    coefficients = model.coef_
    feature_importance = pd.DataFrame({
        'Feature': best_features,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nModel coefficients:")
    print(feature_importance.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Model Coefficients')
    plt.tight_layout()
    plt.show()


# In[38]:


# =================================================================
# 4. FINAL MODEL EVALUATION ON TEST SET
# =================================================================
print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION ON TEST SET")
print("=" * 60)

# Evaluate final model on test set
y_test_pred = tuned_model.predict(X_test_best)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Final model performance on test set:")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"R²: {r2_test:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Complexity')
plt.ylabel('Predicted Complexity')
plt.title('Test Set: Actual vs Predicted Complexity')
plt.tight_layout()
plt.show()

# Visualize prediction errors
errors = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Error analysis by corpus
if 'corpus' in df_features.columns:
    test_indices = X_test.index
    test_df = df_features.loc[test_indices].copy()
    test_df['predicted'] = y_test_pred
    test_df['actual'] = y_test
    test_df['error'] = test_df['actual'] - test_df['predicted']
    test_df['abs_error'] = np.abs(test_df['error'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='corpus', y='abs_error', data=test_df)
    plt.title('Absolute Error by Corpus')
    plt.xlabel('Corpus')
    plt.ylabel('Absolute Error')
    plt.tight_layout()
    plt.show()
    
    corpus_performance = test_df.groupby('corpus').agg({
        'abs_error': ['mean', 'std', 'median'],
        'error': ['mean']
    })
    
    print("\nError by corpus:")
    print(corpus_performance)

# Identify words with largest errors
num_examples = 20
largest_errors = test_df.sort_values('abs_error', ascending=False).head(num_examples)
print(f"\nWords with largest prediction errors:")
for i, (_, row) in enumerate(largest_errors.iterrows(), 1):
    print(f"{i}. '{row['token']}' (corpus: {row['corpus']})")
    print(f"   Actual complexity: {row['actual']:.4f}, Predicted: {row['predicted']:.4f}, Error: {row['error']:.4f}")


# In[39]:


# =================================================================
# 5. SAVE MODEL AND RESULTS
# =================================================================
print("\n" + "=" * 60)
print("SAVING MODEL AND RESULTS")
print("=" * 60)

import joblib

# Save the best model
model_filename = f'data/complexity_prediction_model.joblib'
joblib.dump(tuned_model, model_filename)
print(f"Saved model to {model_filename}")

# Save feature list for future use
feature_filename = f'data/complexity_model_features.txt'
with open(feature_filename, 'w') as f:
    for feature in best_features:
        f.write(f"{feature}\n")
print(f"Saved feature list to {feature_filename}")

# Save summary results
summary = {
    'best_model': best_model_name,
    'feature_group': best_feature_group,
    'num_features': len(best_features),
    'validation_rmse': rmse_val,
    'test_rmse': rmse_test,
    'test_mae': mae_test,
    'test_r2': r2_test
}

# Print final summary
print("\nFinal Model Summary:")
print(f"Model type: {summary['best_model']}")
print(f"Feature group: {summary['feature_group']}")
print(f"Number of features: {summary['num_features']}")
print(f"Validation RMSE: {summary['validation_rmse']:.4f}")
print(f"Test RMSE: {summary['test_rmse']:.4f}")
print(f"Test MAE: {summary['test_mae']:.4f}")
print(f"Test R²: {summary['test_r2']:.4f}")

print("\nTop 10 most important features:")
if 'feature_importance' in locals():
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        if 'Importance' in feature_importance.columns:
            value_col = 'Importance'
        else:
            value_col = 'Coefficient'
        print(f"{i}. {row['Feature']}: {row[value_col]:.4f}")


# In[ ]: