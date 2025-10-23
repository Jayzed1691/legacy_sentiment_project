"""
Utils Module

Utility functions for file loading, text processing, etc.
"""

from sentiment_analyzer.utils.file_loader import (
    load_json_file,
    load_custom_entities,
    load_multiword_expressions,
    load_regex_patterns,
    load_stopwords,
    load_language_data,
)
from sentiment_analyzer.utils.stopwords import StopwordFilter

__all__ = [
    # File loading
    "load_json_file",
    "load_custom_entities",
    "load_multiword_expressions",
    "load_regex_patterns",
    "load_stopwords",
    "load_language_data",
    # Text processing
    "StopwordFilter",
]
