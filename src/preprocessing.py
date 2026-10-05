"""NLP text preprocessing pipeline for BANKING77 dataset."""
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Build stopword set — preserve negation words critical for intent detection
_STOP_WORDS = set(stopwords.words('english'))
_NEGATION_WORDS = {'not', 'no', 'nor', 'don', "don't", 'didn', "didn't",
                   'won', "won't", 'can', "can't", 'couldn', "couldn't",
                   'shouldn', "shouldn't", 'wouldn', "wouldn't",
                   'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                   'aren', "aren't", 'wasn', "wasn't", 'weren', "weren't"}
STOP_WORDS = _STOP_WORDS - _NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Clean and normalize a single text string.
    
    Steps:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Tokenize
        4. Remove stopwords (keeping negation words)
        5. Lemmatize
        6. Remove single-character tokens
    
    Args:
        text: Raw input text string.
    
    Returns:
        Cleaned and normalized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and single-char tokens, then lemmatize
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    
    return ' '.join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply preprocessing to an entire pandas Series.
    
    Args:
        series: pandas Series of raw text strings.
    
    Returns:
        pandas Series of preprocessed text strings.
    """
    return series.apply(preprocess_text)
