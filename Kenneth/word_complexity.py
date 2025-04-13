import re
import numpy as np
import nltk
from nltk.corpus import cmudict, wordnet
from collections import Counter

try:
    nltk.data.find('corpora/cmudict')
    cmu_dict = cmudict.dict()
except:
    nltk.download('cmudict')
    from nltk.corpus import cmudict
    cmu_dict = cmudict.dict()

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
    
class WordComplexityPredictor:
    def __init__(self):
        # Preset frequency mappings for common words
        # These would typically come from a large corpus analysis
        # For simplicity, we're implementing a small subset
        self.frequency_mapping = {
            'the': 7.14, 'of': 6.70, 'and': 6.32, 'to': 6.28, 'a': 6.22, 
            'in': 6.12, 'that': 5.92, 'is': 5.91, 'it': 5.68, 'for': 5.63,
            'computer': 4.2, 'technology': 3.9, 'algorithm': 3.1, 'hypothesis': 3.0,
            'ubiquitous': 2.5, 'ephemeral': 2.3, 'esoteric': 2.0, 'paradigm': 2.8,
            'quantum': 3.2, 'philosophy': 3.4, 'metaphor': 3.1, 'cognition': 2.7,
            'serendipity': 2.2, 'quintessential': 2.1, 'juxtaposition': 2.0, 'idiosyncrasy': 1.9
        }
        
    def count_syllables(self, word):
        """Count syllables in a word using CMU dictionary with fallback."""
        word = word.lower().strip()
        cleaned_word = re.sub(r'[^a-z]', '', word)
        
        if not cleaned_word:
            return 0
            
        # Check for the word in the CMU dictionary
        if cleaned_word in cmu_dict:
            # Count number of digits in the pronunciation (each digit represents a stressed vowel)
            return max(1, len([ph for ph in cmu_dict[cleaned_word][0] if any(c.isdigit() for c in ph)]))
        
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
    
    def count_consonant_clusters(self, word):
        """Count sequences of 2+ consonants"""
        return len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{2,}', word.lower()))
    
    def count_complex_phonemes(self, word):
        """Count complex phoneme combinations"""
        complex_phonemes = ['ph', 'th', 'sh', 'ch', 'wh', 'gh', 'ght', 'sch', 'scr', 'squ', 'kn', 
                           'gn', 'ps', 'pn', 'mn', 'rh']
        word = word.lower()
        return sum(1 for phoneme in complex_phonemes if phoneme in word)
    
    def get_wordnet_features(self, word):
        """Extract features from WordNet"""
        word = word.lower()
        # Clean the word to improve matching
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        # Get synsets
        synsets = wordnet.synsets(clean_word)
        
        # Number of meanings (polysemy)
        num_meanings = len(synsets)
        
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
    
    def char_bigram_rarity(self, token):
        """Measure how 'unusual' the character combinations are"""
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
    
    def get_frequency_score(self, word):
        """Get word frequency score (log10 frequency or estimated)"""
        word = word.lower()
        
        # Check if word is in our frequency mapping
        if word in self.frequency_mapping:
            return self.frequency_mapping[word]
        
        # Estimate frequency based on word properties
        # Shorter words tend to be more common
        length_penalty = max(0, len(word) - 4) * 0.2
        
        # Words with more syllables tend to be less frequent
        syllable_count = self.count_syllables(word)
        syllable_penalty = max(0, syllable_count - 2) * 0.3
        
        # Words with unusual letter combinations tend to be less frequent
        rarity_score = self.char_bigram_rarity(word) * 2
        
        # Base score for unknown words
        base_score = 4.0  # Moderate frequency (scale is roughly 1-7)
        
        estimated_score = base_score - length_penalty - syllable_penalty - rarity_score
        return max(1.0, min(7.0, estimated_score))  # Clamp to reasonable range
    
    def extract_features(self, word):
        """Extract features for word complexity prediction"""
        features = {}
        
        # Basic word features
        features['word_length'] = len(word)
        features['num_syllables'] = self.count_syllables(word)
        
        # Character-level features
        features['num_vowels'] = len(re.findall(r'[aeiou]', word.lower()))
        features['num_consonants'] = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', word.lower()))
        
        if features['word_length'] > 0:
            features['vowel_ratio'] = features['num_vowels'] / features['word_length']
            features['consonant_ratio'] = features['num_consonants'] / features['word_length']
            features['syllables_per_char'] = features['num_syllables'] / features['word_length']
        else:
            features['vowel_ratio'] = 0
            features['consonant_ratio'] = 0
            features['syllables_per_char'] = 0
        
        # Morphological complexity
        features['is_capitalized'] = 1 if word and word[0].isupper() else 0
        features['is_all_caps'] = 1 if word.isupper() and len(word) > 1 else 0
        
        # Phonological features
        features['consonant_clusters'] = self.count_consonant_clusters(word)
        features['num_complex_phonemes'] = self.count_complex_phonemes(word)
        features['is_polysyllabic'] = 1 if features['num_syllables'] > 2 else 0
        
        # Frequency features
        freq_score = self.get_frequency_score(word)
        features['freq_log10'] = freq_score
        features['inverse_frequency'] = 1 / (freq_score + 1)  # Add 1 to avoid division by zero
        
        # WordNet features
        try:
            num_meanings, avg_depth, num_hypernyms, num_hyponyms = self.get_wordnet_features(word)
            features['num_meanings'] = num_meanings
            features['wordnet_depth'] = avg_depth
            features['num_hypernyms'] = num_hypernyms
            features['num_hyponyms'] = num_hyponyms
        except:
            features['num_meanings'] = 0
            features['wordnet_depth'] = 0
            features['num_hypernyms'] = 0
            features['num_hyponyms'] = 0
        
        # Advanced derived features
        features['rare_bigram_ratio'] = self.char_bigram_rarity(word)
        
        # Create combined indices
        features['cognitive_load_index'] = (
            (3 - features['freq_log10'] / 7 * 3) +  # Remap to 0-3 scale
            (features['num_syllables'] / 3) +  # Most words are 1-3 syllables
            (features['word_length'] / 10)  # Most words are under 10 letters
        )
        
        return features
    
    def predict_complexity(self, word):
        """Predict the complexity of a word on a scale from 0-1"""
        features = self.extract_features(word)
        
        # Weighted combination of key features
        # These weights are based on our feature importance analysis
        complexity = (
            0.35 * (1 - features['freq_log10'] / 7) +  # Frequency (inverted)
            0.20 * features['cognitive_load_index'] / 5 +  # Cognitive load
            0.15 * features['syllables_per_char'] +  # Syllable density
            0.10 * (features['num_syllables'] / 5) +  # Syllable count
            0.10 * (features['word_length'] / 15) +  # Word length
            0.05 * features['rare_bigram_ratio'] +  # Unusual letter combinations
            0.05 * (features['is_polysyllabic'] * 0.5)  # Polysyllabic bonus
        )
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, complexity))
    
    def evaluate_complexity(self, word, scale=100):
        """Public method to evaluate word complexity on a 0-100 scale"""
        complexity = self.predict_complexity(word)
        return complexity * scale

def word_complexity_score(word):
    """
    Simple function to get a word complexity score without needing to instantiate the class.
    Returns a score from 0-100, where higher = more complex.
    """
    predictor = WordComplexityPredictor()
    return predictor.evaluate_complexity(word)