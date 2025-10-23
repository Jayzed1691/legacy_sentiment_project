"""
Stopword Filter

Simple stopword filtering for text preprocessing.
Combines standard English stopwords with custom domain-specific stopwords.
"""

import logging
from typing import List, Set, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class StopwordFilter:
    """
    Filters stopwords from text.

    Combines standard English stopwords with custom domain-specific stopwords
    loaded from JSON files.

    Example:
        >>> filter = StopwordFilter(['data/language/custom_stopwords.json'])
        >>> filter.is_stopword('the')
        True
        >>> filter.is_stopword('revenue')
        False
    """

    # Basic English stopwords (minimal set)
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with'
    }

    def __init__(self, custom_stopwords_files: Union[str, List[str]] = None,
                 case_sensitive: bool = False, use_default: bool = True):
        """
        Initialize stopword filter.

        Args:
            custom_stopwords_files: Path(s) to JSON file(s) with custom stopwords
            case_sensitive: Whether to treat stopwords as case-sensitive
            use_default: Whether to include default English stopwords
        """
        self.case_sensitive = case_sensitive
        self.stopwords = set()

        # Add default stopwords if requested
        if use_default:
            # Try to load from spaCy if available
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                self.stopwords = set(nlp.Defaults.stop_words)
                logger.info(f"Loaded {len(self.stopwords)} stopwords from spaCy")
            except (ImportError, OSError):
                # Fall back to basic stopwords
                self.stopwords = self.DEFAULT_STOPWORDS.copy()
                logger.info(f"Using default stopwords ({len(self.stopwords)} words)")

        # Load custom stopwords if provided
        if custom_stopwords_files:
            custom = self._load_custom_stopwords(custom_stopwords_files)
            self.stopwords.update(custom)
            logger.info(f"Total stopwords: {len(self.stopwords)} (including {len(custom)} custom)")

        # Normalize case if not case-sensitive
        if not self.case_sensitive:
            self.stopwords = {word.lower() for word in self.stopwords}

    def _load_custom_stopwords(self, files: Union[str, List[str]]) -> Set[str]:
        """Load custom stopwords from JSON files."""
        from sentiment_analyzer.utils.file_loader import load_stopwords

        stopwords_dict = load_stopwords(files)
        stopwords = set()

        # Extract stopwords from all categories
        for category, words in stopwords_dict.items():
            if isinstance(words, list):
                stopwords.update(words)
            elif isinstance(words, str):
                stopwords.add(words)

        # Normalize case if needed
        if not self.case_sensitive:
            stopwords = {word.lower() for word in stopwords}

        return stopwords

    def is_stopword(self, word: str) -> bool:
        """
        Check if a word is a stopword.

        Args:
            word: Word to check

        Returns:
            True if word is a stopword
        """
        if not self.case_sensitive:
            word = word.lower()
        return word in self.stopwords

    def filter_words(self, words: List[str]) -> List[str]:
        """
        Filter stopwords from a list of words.

        Args:
            words: List of words to filter

        Returns:
            List of words with stopwords removed
        """
        return [word for word in words if not self.is_stopword(word)]

    def filter_text(self, text: str) -> str:
        """
        Filter stopwords from text.

        Args:
            text: Text to filter

        Returns:
            Text with stopwords removed (space-separated)
        """
        words = text.split()
        filtered = self.filter_words(words)
        return ' '.join(filtered)

    def add_stopwords(self, words: Union[str, List[str]]) -> None:
        """
        Add custom stopwords dynamically.

        Args:
            words: Word or list of words to add
        """
        if isinstance(words, str):
            words = [words]

        for word in words:
            if not self.case_sensitive:
                word = word.lower()
            self.stopwords.add(word)

        logger.debug(f"Added {len(words)} stopwords")

    def remove_stopwords(self, words: Union[str, List[str]]) -> None:
        """
        Remove words from stopword list.

        Args:
            words: Word or list of words to remove
        """
        if isinstance(words, str):
            words = [words]

        for word in words:
            if not self.case_sensitive:
                word = word.lower()
            self.stopwords.discard(word)

        logger.debug(f"Removed {len(words)} stopwords")

    def get_stopwords(self) -> Set[str]:
        """Get the current set of stopwords."""
        return self.stopwords.copy()

    def __len__(self) -> int:
        """Return number of stopwords."""
        return len(self.stopwords)

    def __contains__(self, word: str) -> bool:
        """Check if word is a stopword (enables 'word in filter' syntax)."""
        return self.is_stopword(word)
