"""
Pattern Extractor

Extracts patterns from text using regular expressions.
Useful for financial patterns (currency, percentages), time patterns, etc.
"""

import logging
import re
from typing import List, Tuple, Dict, Union

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    Extracts patterns from text using regex matching.

    Supports multiple pattern categories with priority ordering to handle
    overlapping matches.

    Example:
        >>> extractor = PatternExtractor(['data/language/custom_regex_patterns.json'])
        >>> patterns = extractor.extract_patterns("Revenue was $95.5 billion in Q3 2024")
        >>> # Returns: [('$95.5', 'currency', 'Regex', 12, 17),
        >>> #           ('Q3 2024', 'time', 'Regex', 29, 37), ...]
    """

    # Default priority order (highest to lowest)
    DEFAULT_PRIORITY = ['currency_patterns', 'time_patterns', 'financial_patterns']

    def __init__(self, pattern_files: Union[str, List[str]], priority: List[str] = None):
        """
        Initialize pattern extractor with regex patterns.

        Args:
            pattern_files: Path(s) to JSON file(s) containing regex patterns
            priority: Optional list defining category priority order
        """
        from sentiment_analyzer.utils.file_loader import load_regex_patterns
        self.patterns = load_regex_patterns(pattern_files)
        self.priority = priority or self.DEFAULT_PRIORITY

        # Count total patterns
        total = sum(len(v) for v in self.patterns.values())
        logger.info(f"Loaded {total} regex patterns across {len(self.patterns)} categories")

    def extract_patterns(self, text: str) -> List[Tuple[str, str, str, int, int]]:
        """
        Extract all matching patterns from text.

        Args:
            text: Input text to search for patterns

        Returns:
            List of tuples: (matched_text, label, source, start_pos, end_pos)

        Example:
            >>> patterns = extractor.extract_patterns("Growth was 8.5% in Q3")
            >>> # Returns: [('8.5%', 'percentage', 'Regex', 11, 15),
            >>> #           ('Q3', 'quarter', 'Regex', 19, 21)]
        """
        all_matches = []

        # Process categories in priority order first
        for category in self.priority:
            if category in self.patterns:
                self._extract_category_patterns(text, category, all_matches)

        # Process remaining categories
        for category in self.patterns:
            if category not in self.priority:
                self._extract_category_patterns(text, category, all_matches)

        # Sort by position, then by length (longer matches first)
        all_matches.sort(key=lambda x: (x[3], -len(x[0])))

        # Remove overlaps (keep first/longest match at each position)
        return self._remove_overlaps(all_matches)

    def _extract_category_patterns(
        self,
        text: str,
        category: str,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> None:
        """
        Extract all patterns from a category.

        Args:
            text: Text to search in
            category: Pattern category name
            matches: List to append matches to (modified in place)
        """
        for pattern_dict in self.patterns[category]:
            pattern = pattern_dict.get('pattern')
            if not pattern:
                logger.warning(f"Pattern dict missing 'pattern' key: {pattern_dict}")
                continue

            label = pattern_dict.get('label', category)

            try:
                # Find all matches for this pattern
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append((
                        match.group(),
                        label,
                        'Regex',
                        match.start(),
                        match.end()
                    ))
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")

    def _remove_overlaps(
        self,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> List[Tuple[str, str, str, int, int]]:
        """
        Remove overlapping matches, keeping earlier/longer ones.

        Args:
            matches: Sorted list of matches

        Returns:
            List of non-overlapping matches
        """
        non_overlapping = []
        last_end = -1

        for match in matches:
            start = match[3]
            end = match[4]

            # Keep match if it starts at or after the previous match ended
            if start >= last_end:
                non_overlapping.append(match)
                last_end = end

        return non_overlapping

    def add_pattern(self, category: str, pattern: str, label: str = None) -> None:
        """
        Dynamically add a new pattern.

        Args:
            category: Category to add pattern to
            pattern: Regex pattern string
            label: Optional label (defaults to category)

        Example:
            >>> extractor.add_pattern('custom', r'\\bFOO\\d+', 'foo_code')
        """
        if category not in self.patterns:
            self.patterns[category] = []

        self.patterns[category].append({
            'pattern': pattern,
            'label': label or category
        })

        logger.info(f"Added pattern to category '{category}': {pattern}")
