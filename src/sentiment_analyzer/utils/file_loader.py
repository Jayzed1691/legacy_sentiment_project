"""
File Loader Utilities

Utilities for loading JSON configuration files and data dictionaries.
Supports both file paths and optional Streamlit UploadedFile objects.
"""

import json
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional Streamlit support
try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    STREAMLIT_AVAILABLE = True
except ImportError:
    UploadedFile = None
    STREAMLIT_AVAILABLE = False


def load_json_file(file: Union[str, Path, Any]) -> Dict[str, Any]:
    """
    Load a JSON file from a file path or UploadedFile object.

    Args:
        file: File path (str/Path) or Streamlit UploadedFile object

    Returns:
        Loaded JSON data as dictionary, or empty dict on error

    Example:
        >>> data = load_json_file('data/language/custom_entities.json')
        >>> print(data.keys())
    """
    try:
        if isinstance(file, (str, Path)):
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif STREAMLIT_AVAILABLE and isinstance(file, UploadedFile):
            return json.loads(file.read().decode('utf-8'))
        else:
            raise ValueError(f"Unsupported file type: {type(file)}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file}: {e}")
        return {}
    except FileNotFoundError:
        logger.error(f"File not found: {file}")
        return {}
    except Exception as e:
        logger.error(f"Error loading file {file}: {e}")
        return {}


def load_custom_entities(
    files: Union[str, Path, List[Union[str, Path]]]
) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
    """
    Load custom entities from one or more JSON files.

    File format:
    {
        "COMPANY": [
            {"term": "Apple Inc.", "variations": ["Apple", "AAPL"]},
            "Microsoft"
        ],
        "FINANCIAL_VOCABULARY": [...]
    }

    Args:
        files: Single file path or list of file paths

    Returns:
        Dictionary mapping entity types to lists of entities

    Example:
        >>> entities = load_custom_entities('data/language/custom_entities.json')
        >>> print(entities['COMPANY'][0])
        {'term': 'Apple Inc.', 'variations': ['Apple', 'AAPL']}
    """
    if isinstance(files, (str, Path)):
        files = [files]

    custom_entities: Dict[str, List[Union[str, Dict[str, Any]]]] = {}

    for file in files:
        try:
            content = load_json_file(file)
            if not isinstance(content, dict):
                logger.warning(f"Invalid format in {file}. Expected dictionary.")
                continue

            for entity_type, entities in content.items():
                if entity_type not in custom_entities:
                    custom_entities[entity_type] = []

                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict):
                            # Standardize dict format
                            if 'term' in entity or 'full_name' in entity:
                                custom_entities[entity_type].append(entity)
                            else:
                                logger.warning(f"Entity dict missing 'term' field: {entity}")
                        elif isinstance(entity, str):
                            # Convert string to dict format
                            custom_entities[entity_type].append({
                                'term': entity,
                                'category': [entity_type.lower()]
                            })
                        else:
                            logger.warning(f"Invalid entity format: {entity}")
                else:
                    logger.warning(f"Invalid format for {entity_type}. Expected list.")

        except Exception as e:
            logger.error(f"Error loading entities from {file}: {e}")

    # Deduplicate entities (except PERSON category which may have duplicates)
    for entity_type in custom_entities:
        if entity_type != 'PERSON':
            # Deduplicate by term
            seen = {}
            for entity in custom_entities[entity_type]:
                if isinstance(entity, dict):
                    term = entity.get('term', entity.get('full_name'))
                    if term and term not in seen:
                        seen[term] = entity
                elif isinstance(entity, str) and entity not in seen:
                    seen[entity] = entity
            custom_entities[entity_type] = list(seen.values())

    logger.info(f"Loaded {sum(len(v) for v in custom_entities.values())} entities "
               f"across {len(custom_entities)} categories")

    return custom_entities


def load_multiword_expressions(
    files: Union[str, Path, List[Union[str, Path]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load multi-word expressions from one or more JSON files.

    File format:
    {
        "FINANCIAL_VOCABULARY": [
            {"term": "earnings per share", "category": ["financial"]},
            {"term": "year over year", "variations": ["YoY", "y-o-y"]}
        ]
    }

    Args:
        files: Single file path or list of file paths

    Returns:
        Dictionary mapping categories to lists of MWE dictionaries

    Example:
        >>> mwes = load_multiword_expressions('data/language/custom_mwe.json')
        >>> print(mwes['FINANCIAL_VOCABULARY'][0])
        {'term': 'earnings per share', 'category': ['financial']}
    """
    if isinstance(files, (str, Path)):
        files = [files]

    multi_word_entries: Dict[str, List[Dict[str, Any]]] = {}

    for file in files:
        try:
            content = load_json_file(file)
            if not isinstance(content, dict):
                logger.warning(f"Invalid format in {file}. Expected dictionary.")
                continue

            for category, entries in content.items():
                if category not in multi_word_entries:
                    multi_word_entries[category] = []

                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and 'term' in entry:
                            multi_word_entries[category].append(entry)
                        else:
                            logger.warning(f"Invalid MWE entry: {entry}")
                else:
                    logger.warning(f"Invalid format for category {category}")

        except Exception as e:
            logger.error(f"Error loading MWEs from {file}: {e}")

    # Deduplicate by term
    for category in multi_word_entries:
        seen = {entry['term']: entry for entry in multi_word_entries[category]}
        multi_word_entries[category] = list(seen.values())

    logger.info(f"Loaded {sum(len(v) for v in multi_word_entries.values())} "
               f"multi-word expressions across {len(multi_word_entries)} categories")

    return multi_word_entries


def load_regex_patterns(
    files: Union[str, Path, List[Union[str, Path]]]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Load regex patterns from one or more JSON files.

    File format:
    {
        "currency_patterns": [
            {"pattern": "\\$\\d+(\\.\\d{2})?", "label": "USD"},
            {"pattern": "€\\d+(\\.\\d{2})?", "label": "EUR"}
        ],
        "time_patterns": [...]
    }

    Args:
        files: Single file path or list of file paths

    Returns:
        Dictionary mapping pattern categories to lists of pattern dictionaries

    Example:
        >>> patterns = load_regex_patterns('data/language/custom_regex_patterns.json')
        >>> print(patterns['currency_patterns'][0])
        {'pattern': '\\$\\d+(\\.\\d{2})?', 'label': 'USD'}
    """
    if isinstance(files, (str, Path)):
        files = [files]

    patterns: Dict[str, List[Dict[str, str]]] = {}

    for file in files:
        try:
            content = load_json_file(file)
            if not isinstance(content, dict):
                logger.warning(f"Invalid format in {file}. Expected dictionary.")
                continue

            for category, category_patterns in content.items():
                if category not in patterns:
                    patterns[category] = []

                if isinstance(category_patterns, list):
                    patterns[category].extend(category_patterns)
                else:
                    logger.warning(f"Invalid format for category {category}")

        except Exception as e:
            logger.error(f"Error loading regex patterns from {file}: {e}")

    logger.info(f"Loaded {sum(len(v) for v in patterns.values())} "
               f"regex patterns across {len(patterns)} categories")

    return patterns


def load_stopwords(
    files: Union[str, Path, List[Union[str, Path]]]
) -> Dict[str, List[str]]:
    """
    Load custom stopwords from one or more JSON files.

    File format:
    {
        "custom_stopwords": ["the", "a", "an", ...]
    }

    Or just a list:
    ["the", "a", "an", ...]

    Args:
        files: Single file path or list of file paths

    Returns:
        Dictionary with "stopwords" key containing deduplicated list

    Example:
        >>> stopwords = load_stopwords('data/language/custom_stops.json')
        >>> print(len(stopwords['stopwords']))
        1723
    """
    if isinstance(files, (str, Path)):
        files = [files]

    all_stopwords: List[str] = []

    for file in files:
        try:
            content = load_json_file(file)

            if isinstance(content, dict) and 'custom_stopwords' in content:
                all_stopwords.extend(content['custom_stopwords'])
            elif isinstance(content, list):
                all_stopwords.extend(content)
            else:
                logger.warning(f"Invalid stopwords format in {file}")

        except Exception as e:
            logger.error(f"Error loading stopwords from {file}: {e}")

    # Deduplicate
    all_stopwords = list(set(all_stopwords))

    logger.info(f"Loaded {len(all_stopwords)} unique stopwords")

    return {"stopwords": all_stopwords}


def load_language_data(
    files: Union[str, Path, List[Union[str, Path]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load language data (modifiers, uncertainty words, etc.) from JSON files.

    File format:
    {
        "quantitative_modifiers": [
            {"term": "approximately", "category": ["modifiers", "quantitative"], "pos": ["ADV"]},
            {"term": "roughly", "category": ["modifiers", "quantitative"], "pos": ["ADV"]}
        ],
        "uncertainty_words": [...]
    }

    Args:
        files: Single file path or list of file paths

    Returns:
        Dictionary mapping categories to lists of term dictionaries

    Example:
        >>> lang_data = load_language_data('data/language/language_data.json')
        >>> print(lang_data['quantitative_modifiers'][0])
        {'term': 'approximately', 'category': [...], 'pos': ['ADV']}
    """
    if isinstance(files, (str, Path)):
        files = [files]

    language_data: Dict[str, List[Dict[str, Any]]] = {}

    for file in files:
        try:
            content = load_json_file(file)
            if not isinstance(content, dict):
                logger.warning(f"Invalid format in {file}. Expected dictionary.")
                continue

            for category, terms in content.items():
                if category not in language_data:
                    language_data[category] = []

                if isinstance(terms, list):
                    for term in terms:
                        if isinstance(term, dict):
                            # Ensure required fields
                            if 'term' not in term:
                                logger.warning(f"Term missing 'term' field: {term}")
                                continue

                            # Standardize
                            standardized = {
                                'term': term['term'],
                                'category': term.get('category', [category.lower()]),
                                'pos': term.get('pos', ['ADV']),
                            }

                            # Add optional fields
                            if 'match_type' in term:
                                standardized['match_type'] = term['match_type']
                            elif ' ' in term['term']:
                                standardized['match_type'] = 'phrase'

                            language_data[category].append(standardized)

                        elif isinstance(term, str):
                            # Convert string to dict
                            language_data[category].append({
                                'term': term,
                                'category': [category.lower()],
                                'pos': ['ADV']
                            })
                else:
                    logger.warning(f"Invalid format for category {category}")

        except Exception as e:
            logger.error(f"Error loading language data from {file}: {e}")

    # Deduplicate by term
    for category in language_data:
        seen = {term['term']: term for term in language_data[category]}
        language_data[category] = list(seen.values())

    logger.info(f"Loaded {sum(len(v) for v in language_data.values())} "
               f"language terms across {len(language_data)} categories")

    return language_data
