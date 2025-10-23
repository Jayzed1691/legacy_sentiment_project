"""
Multi-Word Expression Extractor

Extracts multi-word expressions (MWEs) from text, such as "year over year",
"earnings per share", "chief financial officer", etc.
"""

import logging
from typing import List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)


class MultiwordExtractor:
    """
    Extracts multi-word expressions from text based on predefined dictionaries.

    Similar to EntityExtractor but specifically for multi-word phrases that
    should be treated as single units.

    Example:
        >>> extractor = MultiwordExtractor(['data/language/custom_mwe.json'])
        >>> mwes = extractor.extract_multiword_expressions("year over year growth")
        >>> # Returns: [('year over year', 'FINANCIAL', 'MWE', 0, 14), ...]
    """

    def __init__(self, mwe_files: Union[str, List[str]]):
        """
        Initialize MWE extractor with dictionaries.

        Args:
            mwe_files: Path(s) to JSON file(s) containing MWE definitions
        """
        from sentiment_analyzer.utils.file_loader import load_multiword_expressions
        self.mwe_entries = load_multiword_expressions(mwe_files)
        logger.info(f"Loaded {sum(len(v) for v in self.mwe_entries.values())} "
                   f"multi-word expressions across {len(self.mwe_entries)} categories")

    def extract_multiword_expressions(self, text: str) -> List[Tuple[str, str, str, int, int]]:
        """
        Extract all matching multi-word expressions from text.

        Args:
            text: Input text to search for MWEs

        Returns:
            List of tuples: (mwe_text, category, source, start_pos, end_pos)
        """
        all_matches = []

        # Search for all MWE terms across all categories
        for category, terms in self.mwe_entries.items():
            for term in terms:
                if isinstance(term, dict) and 'term' in term:
                    self._add_mwe_to_matches(text, term['term'], category, all_matches)
                elif isinstance(term, str):
                    self._add_mwe_to_matches(text, term, category, all_matches)

        # Sort by position, then by length (longer matches first)
        all_matches.sort(key=lambda x: (x[3], -len(x[0])))

        # Remove overlaps
        return self._remove_overlaps(all_matches)

    def _add_mwe_to_matches(
        self,
        text: str,
        mwe: str,
        category: str,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> None:
        """Find all occurrences of an MWE in text."""
        start = 0
        while True:
            start = text.lower().find(mwe.lower(), start)
            if start == -1:
                break

            end = start + len(mwe)

            # Check word boundaries
            if self._is_word_boundary(text, start, end):
                matches.append((text[start:end], category, 'MWE', start, end))

            start = end

    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        """Check if match is at word boundaries."""
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end == len(text) or not text[end].isalnum()
        return before_ok and after_ok

    def _remove_overlaps(
        self,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> List[Tuple[str, str, str, int, int]]:
        """Remove overlapping matches."""
        final_mwes = []

        for mwe, category, source, start, end in matches:
            is_overlap = any(
                self._is_complete_overlap(start, end, other_start, other_end)
                for _, _, _, other_start, other_end in final_mwes
            )

            if not is_overlap:
                final_mwes.append((mwe, category, source, start, end))

        return final_mwes

    def _is_complete_overlap(
        self,
        start1: int,
        end1: int,
        start2: int,
        end2: int
    ) -> bool:
        """Check if range1 is completely within range2."""
        return (start1 >= start2 and
                end1 <= end2 and
                (start1, end1) != (start2, end2))
