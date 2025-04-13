import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from sklearn.ensemble import RandomForestRegressor

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    from nltk.corpus import cmudict
    cmu_dict = cmudict.dict()
except:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('cmudict')
    from nltk.corpus import cmudict
    cmu_dict = cmudict.dict()

def count_syllables(word):
    word = word.lower().strip()
    word = re.sub(r'[^a-z]', '', word)
    
    if not word:
        return 0
        
    if word in cmu_dict:
        return max(1, len([ph for ph in cmu_dict[word][0] if any(c.isdigit() for c in ph)]))
    
    exceptions = {
        'coed': 2, 'loved': 1, 'lives': 1, 'moved': 1, 'lived': 1, 'hated': 2,
        'wanted': 2, 'ended': 2, 'rated': 2, 'used': 1, 'caused': 1, 'forced': 1,
        'hoped': 1, 'deserved': 2, 'named': 1, 'the': 1, 'a': 1, 'i': 1, 'an': 1,
        'area': 3, 'aria': 3, 'eye': 1, 'ate': 1, 'once': 1, 'are': 1
    }
    
    if word in exceptions:
        return exceptions[word]
    
    if len(word) <= 1:
        return 1
        
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    if len(word) >= 2:
        if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if word.endswith('y') and len(word) > 1 and word[-2] not in vowels:
            count += 1
        if (word.endswith('es') or word.endswith('ed')) and len(word) > 2 and word[-3] not in vowels:
            count -= 1
    
    return max(1, count)

def count_text_syllables(text):
    if not text or not isinstance(text, str):
        return 0
    words = re.findall(r'\b\w+\b', text)
    return sum(count_syllables(word) for word in words)

def get_sentence_length_variance(text):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return np.var(lengths)

class SentenceComplexityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.is_trained = False
    
    def extract_features(self, text):
        features = {}
        
        # Basic text stats
        words = re.findall(r'\b\w+\b', text)
        features['word_count'] = len(words)
        
        char_count = len(text.replace(" ", ""))
        features['char_count'] = char_count
        
        features['avg_word_length'] = char_count / max(1, features['word_count'])
        
        sentences = sent_tokenize(text)
        features['sentence_count'] = len(sentences)
        
        features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])
        features['chars_per_sentence'] = char_count / max(1, features['sentence_count'])
        
        # Vocabulary complexity
        unique_words = set(word.lower() for word in words)
        features['unique_word_count'] = len(unique_words)
        features['type_token_ratio'] = features['unique_word_count'] / max(1, features['word_count'])
        
        # Syllable features
        syllable_count = count_text_syllables(text)
        features['syllable_count'] = syllable_count
        features['syllables_per_word'] = syllable_count / max(1, features['word_count'])
        
        # Count polysyllabic words (3+ syllables)
        polysyllable_count = sum(1 for word in words if count_syllables(word) > 2)
        features['polysyllable_count'] = polysyllable_count
        features['polysyllable_ratio'] = polysyllable_count / max(1, features['word_count']) * 100
        
        # Long Words (6+ characters)
        long_word_count = sum(1 for word in words if len(word) >= 6)
        features['long_word_count'] = long_word_count
        features['long_word_ratio'] = long_word_count / max(1, features['word_count']) * 100
        
        # Sentence structure
        comma_count = text.count(',')
        features['comma_count'] = comma_count
        features['commas_per_sentence'] = comma_count / max(1, features['sentence_count'])
        
        punct_count = sum(1 for char in text if char in string.punctuation)
        features['punct_count'] = punct_count
        features['punct_density'] = punct_count / max(1, features['word_count'])
        
        features['sentence_length_variance'] = get_sentence_length_variance(text)
        
        # Dialogue features
        dialogue_marks = text.count('"') + text.count('"') + text.count('"')
        features['dialogue_marks'] = dialogue_marks
        features['dialogue_density'] = dialogue_marks / max(1, features['sentence_count'])
        
        # Technical term density
        technical_term_count = sum(1 for word in words if len(word) > 8)
        features['technical_term_density'] = technical_term_count / max(1, features['word_count']) * 100
        
        # Calculate simple readability metrics
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            # Flesch Reading Ease
            features['flesch_reading_ease'] = 206.835 - (1.015 * features['avg_sentence_length']) - (84.6 * features['syllables_per_word'])
            
            # Flesch-Kincaid Grade Level
            features['flesch_kincaid_grade_level'] = (0.39 * features['avg_sentence_length']) + (11.8 * features['syllables_per_word']) - 15.59
            
            # SMOG Index
            features['smog_readability'] = 1.043 * np.sqrt(polysyllable_count * (30 / max(1, features['sentence_count']))) + 3.1291
            
            # Automated Readability Index
            features['automated_readability_index'] = 4.71 * features['avg_word_length'] + 0.5 * features['avg_sentence_length'] - 21.43
        else:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade_level'] = 0
            features['smog_readability'] = 0
            features['automated_readability_index'] = 0
        
        return features

    def train(self, texts, complexity_scores):
        features_list = []
        for text in texts:
            features = self.extract_features(text)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        self.model.fit(features_df, complexity_scores)
        self.is_trained = True
        
        # Keep track of feature names for prediction
        self.feature_names = features_df.columns.tolist()
        
        return self
    
    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet!")
        
        # Extract features for the new text
        features = self.extract_features(text)
        features_df = pd.DataFrame([features])
        
        # Ensure features match those used during training
        missing_cols = set(self.feature_names) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0  # Default value for missing features
        
        # Select and order columns to match training data
        features_df = features_df[self.feature_names]
        
        # Make prediction
        complexity_score = self.model.predict(features_df)[0]
        
        return complexity_score

    def evaluate_complexity(self, text):
        """
        Public method to evaluate text complexity on a 0-100 scale
        (0 = easiest, 100 = most complex)
        """
        if self.is_trained:
            bt_easiness = self.predict(text)
            # Convert BT Easiness (higher = easier) to complexity (higher = harder)
            # BT Easiness typically ranges from around -3 (hard) to +3 (easy)
            complexity = 50 - (bt_easiness * 10)  # Scale and invert
            return max(0, min(100, complexity))  # Clamp to 0-100 range
        else:
            # Fallback to a simpler scoring if model isn't trained
            features = self.extract_features(text)
            
            # Simple weighted scoring based on key features
            score = (
                0.30 * (features['syllables_per_word'] * 30) +
                0.25 * (features['polysyllable_ratio'] / 2) +
                0.20 * (features['avg_sentence_length'] / 2) +
                0.15 * (features['avg_word_length'] * 10) +
                0.10 * (min(20, features['sentence_count']) / 20 * 30)
            )
            
            return min(100, max(0, score))

def sentence_complexity_score(text):
    """
    Simple complexity scoring function that doesn't require a trained model.
    Returns a score from 0-100, where higher = more complex.
    """
    # Basic text stats
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    if word_count == 0:
        return 50  # Default score for empty text
    
    char_count = len(text.replace(" ", ""))
    avg_word_length = char_count / word_count
    
    sentences = sent_tokenize(text)
    sentence_count = max(1, len(sentences))
    
    avg_sentence_length = word_count / sentence_count
    
    # Syllable features
    syllable_count = count_text_syllables(text)
    syllables_per_word = syllable_count / word_count
    
    # Polysyllabic words (3+ syllables)
    polysyllable_count = sum(1 for word in words if count_syllables(word) > 2)
    polysyllable_ratio = polysyllable_count / word_count * 100
    
    # Compute the score (higher = more complex)
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