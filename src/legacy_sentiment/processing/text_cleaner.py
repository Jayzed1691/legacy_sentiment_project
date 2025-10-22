# text_cleaner.py

from typing import List, Set, Optional
from data_types import Token
import logging

logger = logging.getLogger(__name__)

def clean_matched_text(
    text: str,
    matches: List[Token],
    preserve_punct: Optional[Set[str]] = None,
    use_lemma: bool = True,
    lowercase: bool = True
) -> str:
    """
    Clean text using pre-processed matches from unified_matcher.
    
    Args:
        text: Original text to clean
        matches: List of Token objects from unified_matcher
        preserve_punct: Set of punctuation marks to preserve (defaults to {'.', '!', '?', '%'})
        use_lemma: Whether to use lemmatized forms for non-entity tokens
        lowercase: Whether to convert non-entity tokens to lowercase
    
    Returns:
        Cleaned text string with preserved entities and specified formatting
    """
    preserve_punct = preserve_punct or {'.', '!', '?', '%'}
    cleaned_words = []
    
    def is_punctuation(s: str) -> bool:
        """Check if a string consists entirely of non-alphanumeric characters."""
        return all(not c.isalnum() for c in s)
    
    for match in sorted(matches, key=lambda x: x.start):
        token_text = match.original_text.strip()
        
        # Handle entities and MWEs - preserve original form
        if match.source in {'Entity', 'CustomEntity', 'MWE', 'Regex'}:
            cleaned_words.append(token_text)
            continue
            
        # Handle punctuation
        if is_punctuation(token_text):
            if token_text in preserve_punct:
                if cleaned_words and not is_punctuation(cleaned_words[-1]):
                    cleaned_words[-1] = cleaned_words[-1].rstrip()
                cleaned_words.append(token_text)
            continue
        
        # Handle stopwords and regular tokens
        if match.source in {'custom', 'spacy'} and match.label == 'Stopword':
            word = token_text.lower() if lowercase else token_text
        else:
            word = match.lemma if use_lemma and match.lemma else token_text
            if lowercase:
                word = word.lower()
                
        cleaned_words.append(word)
    
    return " ".join(cleaned_words).strip()

def clean_matched_texts(
    texts: List[str],
    matches_list: List[List[Token]],
    **kwargs
) -> List[str]:
    """
    Clean multiple texts using their corresponding matches.
    
    Args:
        texts: List of original texts
        matches_list: List of match lists (one per text)
        **kwargs: Additional arguments passed to clean_matched_text
    
    Returns:
        List of cleaned text strings
    """
    if len(texts) != len(matches_list):
        raise ValueError("Number of texts must match number of match lists")
        
    return [
        clean_matched_text(text, matches, **kwargs)
        for text, matches in zip(texts, matches_list)
    ]