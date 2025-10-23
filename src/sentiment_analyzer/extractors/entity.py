"""
Entity Extractor

Extracts custom entities from text using dictionary-based matching.
Supports term variations and handles overlapping matches.
"""

import logging
from typing import List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts custom entities from text based on predefined dictionaries.

    Supports:
    - Dictionary terms with variations (e.g., "CFO", "Chief Financial Officer")
    - Word boundary checking
    - Overlap resolution (prioritizes longer matches)
    - Multiple entity categories

    Example:
        >>> extractor = EntityExtractor(['data/language/custom_entities.json'])
        >>> entities = extractor.extract_entities("Apple Inc. reported strong earnings")
        >>> # Returns: [('Apple Inc.', 'COMPANY', 'Custom Entity', 0, 10), ...]
    """

    def __init__(self, entities_files: Union[str, List[str]]):
        """
        Initialize entity extractor with custom entity dictionaries.

        Args:
            entities_files: Path(s) to JSON file(s) containing entity definitions
        """
        from sentiment_analyzer.utils.file_loader import load_custom_entities
        self.entities = load_custom_entities(entities_files)
        logger.info(f"Loaded {sum(len(terms) for terms in self.entities.values())} entities "
                   f"across {len(self.entities)} categories")

    def extract_entities(self, text: str) -> List[Tuple[str, str, str, int, int]]:
        """
        Extract all matching entities from text.

        Args:
            text: Input text to search for entities

        Returns:
            List of tuples: (entity_text, category, source, start_pos, end_pos)

        Example:
            >>> entities = extractor.extract_entities("Q3 revenue was $95.5 billion")
            >>> # Returns: [('Q3', 'TIME_PERIOD', 'Custom Entity', 0, 2),
            >>> #           ('revenue', 'FINANCIAL', 'Custom Entity', 3, 10), ...]
        """
        all_matches = []

        # Search for all entity terms across all categories
        for category, terms in self.entities.items():
            for term in terms:
                if isinstance(term, dict):
                    # Handle dictionary format with variations
                    entity_term = term.get('term') or term.get('full_name')
                    if entity_term:
                        self._add_entity_to_matches(text, entity_term, category, all_matches)

                    # Add variations if they don't overlap with full name
                    for variation in term.get('variations', []):
                        if not self._overlaps_with_full_name(text, variation, entity_term):
                            self._add_entity_to_matches(text, variation, category, all_matches)

                elif isinstance(term, str):
                    # Handle simple string format
                    self._add_entity_to_matches(text, term, category, all_matches)

        # Sort by position, then by length (longer matches first)
        all_matches.sort(key=lambda x: (x[3], -len(x[0])))

        # Remove overlapping matches (keep longer/earlier ones)
        return self._remove_overlaps(all_matches)

    def _add_entity_to_matches(
        self,
        text: str,
        entity: str,
        category: str,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> None:
        """
        Find all occurrences of an entity in text and add to matches list.

        Args:
            text: Text to search in
            entity: Entity term to find
            category: Category label for the entity
            matches: List to append matches to (modified in place)
        """
        start = 0
        while True:
            # Case-insensitive search
            start = text.lower().find(entity.lower(), start)
            if start == -1:
                break

            end = start + len(entity)

            # Check for word boundaries
            if self._is_word_boundary(text, start, end):
                # Preserve original case from text
                matches.append((text[start:end], category, 'Custom Entity', start, end))

            start = end

    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        """
        Check if match is at word boundaries (not part of larger word).

        Args:
            text: Full text
            start: Start position of match
            end: End position of match

        Returns:
            True if match is at word boundaries
        """
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end == len(text) or not text[end].isalnum()
        return before_ok and after_ok

    def _remove_overlaps(
        self,
        matches: List[Tuple[str, str, str, int, int]]
    ) -> List[Tuple[str, str, str, int, int]]:
        """
        Remove overlapping matches, keeping the first/longest ones.

        Args:
            matches: Sorted list of matches

        Returns:
            List of non-overlapping matches
        """
        final_entities = []

        for entity, category, source, start, end in matches:
            # Check if this match completely overlaps with any kept match
            is_overlap = any(
                self._is_complete_overlap(start, end, other_start, other_end)
                for _, _, _, other_start, other_end in final_entities
            )

            if not is_overlap:
                final_entities.append((entity, category, source, start, end))

        return final_entities

    def _is_complete_overlap(
        self,
        start1: int,
        end1: int,
        start2: int,
        end2: int
    ) -> bool:
        """
        Check if range1 is completely within range2.

        Args:
            start1, end1: First range
            start2, end2: Second range

        Returns:
            True if range1 is completely contained in range2 (and they're not identical)
        """
        return (start1 >= start2 and
                end1 <= end2 and
                (start1, end1) != (start2, end2))

    def _overlaps_with_full_name(
        self,
        text: str,
        variation: str,
        full_name: str
    ) -> bool:
        """
        Check if a variation overlaps with the full name in the text.

        This prevents extracting both "CFO" and "Chief Financial Officer" when
        they refer to the same occurrence.

        Args:
            text: Full text
            variation: Shorter variation (e.g., "CFO")
            full_name: Full term (e.g., "Chief Financial Officer")

        Returns:
            True if the variation overlaps with full name in text
        """
        variation_start = text.lower().find(variation.lower())
        full_name_start = text.lower().find(full_name.lower())

        if variation_start == -1 or full_name_start == -1:
            return False

        variation_end = variation_start + len(variation)
        full_name_end = full_name_start + len(full_name)

        # Check if ranges overlap
        return (
            (variation_start <= full_name_start < variation_end) or
            (variation_start < full_name_end <= variation_end) or
            (full_name_start <= variation_start < full_name_end)
        )
