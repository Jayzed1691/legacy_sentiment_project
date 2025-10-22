#!/usr/bin/env python3
"""NLTK-powered preprocessing helpers integrated with custom utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
from nltk.tree import Tree

from legacy_sentiment.utils.custom_file_utils import (
        load_custom_entities,
        load_custom_stopwords,
        load_json_file,
        load_multi_word_entries,
        load_regex_patterns,
)

logger = logging.getLogger(__name__)

_REQUIRED_NLTK_RESOURCES: Tuple[Tuple[str, str], ...] = (
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("corpora/words", "words"),
)


def _ensure_nltk_resources() -> None:
        """Download required NLTK resources when they are missing."""
        for resource_path, resource_name in _REQUIRED_NLTK_RESOURCES:
                try:
                        nltk.data.find(resource_path)
                except LookupError:  # pragma: no cover - defensive download
                        nltk.download(resource_name, quiet=True)


_ensure_nltk_resources()


def _coerce_sequence(value: Any) -> Tuple[str, ...]:
        if value is None:
                return tuple()
        if isinstance(value, str):
                return (value,)
        return tuple(str(item) for item in value)


@dataclass
class PreprocessingConfig:
        """Configuration options that control the preprocessing pipeline."""

        language: str = "english"
        clean: bool = True
        remove_stopwords: bool = True
        lemmatize: bool = True
        preserve_financial_terms: bool = True
        lowercase_entities: bool = False
        custom_stopwords_files: Tuple[str, ...] = field(default_factory=tuple)
        custom_entities_files: Tuple[str, ...] = field(default_factory=tuple)
        regex_pattern_files: Tuple[str, ...] = field(default_factory=tuple)
        multi_word_expression_files: Tuple[str, ...] = field(default_factory=tuple)
        extras: Dict[str, Any] = field(default_factory=dict)

        @classmethod
        def from_mapping(cls, data: Mapping[str, Any]) -> "PreprocessingConfig":
                known_keys = {
                        "language",
                        "clean",
                        "remove_stopwords",
                        "lemmatize",
                        "preserve_financial_terms",
                        "lowercase_entities",
                        "custom_stopwords_files",
                        "custom_entities_files",
                        "regex_pattern_files",
                        "multi_word_expression_files",
                }
                config_kwargs: Dict[str, Any] = {}
                for key in known_keys:
                        if key.endswith("_files"):
                                config_kwargs[key] = _coerce_sequence(data.get(key))
                        elif key in data:
                                config_kwargs[key] = data[key]
                extras = {key: value for key, value in data.items() if key not in known_keys}
                config_kwargs.setdefault("extras", extras)
                return cls(**config_kwargs)

        @classmethod
        def from_file(cls, config_path: str) -> "PreprocessingConfig":
                path = Path(config_path)
                if not path.exists():
                        raise FileNotFoundError(f"Preprocessing configuration not found: {config_path}")
                payload = load_json_file(path.as_posix())
                if not isinstance(payload, Mapping):
                        raise ValueError("Preprocessing configuration must be a JSON object")
                return cls.from_mapping(payload)


@dataclass
class PreprocessingOutput:
        """Container describing the outcome of a preprocessing run."""

        processed_text: str
        tokens: List[str]
        named_entities: List[str]
        custom_patterns: Dict[str, List[str]]

        def as_dict(self) -> Dict[str, Any]:
                return {
                        "processed_text": self.processed_text,
                        "tokens": self.tokens,
                        "named_entities": self.named_entities,
                        "custom_patterns": self.custom_patterns,
                }


class TextPreprocessor:
        """High-level API around the enhanced preprocessing pipeline."""

        def __init__(
                self,
                *,
                language: str,
                stopword_overrides: Iterable[str],
                custom_entities: Mapping[str, Sequence[Mapping[str, Any]]],
                regex_patterns: Mapping[str, Sequence[Mapping[str, Any]]],
                multi_word_entries: Mapping[str, Sequence[Mapping[str, Any]]],
                config: PreprocessingConfig,
        ) -> None:
                self.config = config
                self.language = language
                self.stop_words = self._load_stopwords(language, stopword_overrides)
                self.custom_entities = custom_entities
                self.financial_terms = self._extract_financial_terms(custom_entities)
                self.named_entity_terms = self._collect_named_entities(custom_entities)
                self.lemmatizer = WordNetLemmatizer()
                self.mwe_tokenizer = self._build_mwe_tokenizer(multi_word_entries, self.named_entity_terms)
                self._compiled_patterns = self._compile_patterns(regex_patterns)

        @classmethod
        def from_config(cls, config: PreprocessingConfig) -> "TextPreprocessor":
                stopwords_payload: Dict[str, List[str]] = {}
                if config.custom_stopwords_files:
                        stopwords_payload = load_custom_stopwords(list(config.custom_stopwords_files))
                custom_entities: Dict[str, List[Mapping[str, Any]]] = {}
                if config.custom_entities_files:
                        custom_entities = load_custom_entities(list(config.custom_entities_files))
                multi_word_entries: Dict[str, List[Mapping[str, Any]]] = {}
                if config.multi_word_expression_files:
                        multi_word_entries = load_multi_word_entries(list(config.multi_word_expression_files))
                regex_patterns: Dict[str, List[Mapping[str, Any]]] = {}
                if config.regex_pattern_files:
                        regex_patterns = load_regex_patterns(list(config.regex_pattern_files))
                return cls(
                        language=config.language,
                        stopword_overrides=stopwords_payload.get("stopwords", []),
                        custom_entities=custom_entities,
                        regex_patterns=regex_patterns,
                        multi_word_entries=multi_word_entries,
                        config=config,
                )

        @classmethod
        def from_config_path(cls, config_path: str) -> "TextPreprocessor":
                return cls.from_config(PreprocessingConfig.from_file(config_path))

        def preprocess(self, text: str) -> PreprocessingOutput:
                        patterns = self.extract_custom_patterns(text)
                        working_text = text
                        if self.config.clean:
                                working_text = self.clean_text(working_text)
                        tokens = self.tokenize_text(working_text)
                        if self.config.remove_stopwords:
                                tokens = self.remove_stopwords(tokens)
                        named_entities = self.identify_named_entities(tokens)
                        if self.config.lemmatize:
                                tokens = self.conditional_lemmatization(tokens, named_entities)
                        processed_text = " ".join(tokens)
                        return PreprocessingOutput(
                                processed_text=processed_text,
                                tokens=tokens,
                                named_entities=named_entities,
                                custom_patterns=patterns,
                        )

        def clean_text(self, text: str) -> str:
                text = self._remove_html_tags(text)
                text = self._normalize_whitespace(text)
                if not self.config.lowercase_entities:
                        text = self._conditional_lowercase(text)
                else:
                        text = text.lower()
                text = self._remove_punctuation(text)
                return self._normalize_whitespace(text)

        def tokenize_text(self, text: str) -> List[str]:
                tokens = word_tokenize(text)
                return self.mwe_tokenizer.tokenize(tokens)

        def remove_stopwords(self, tokens: Sequence[str]) -> List[str]:
                preserved_terms = self.financial_terms if self.config.preserve_financial_terms else set()
                filtered_tokens: List[str] = []
                for token in tokens:
                        lower = token.lower()
                        if lower in preserved_terms:
                                filtered_tokens.append(token)
                                continue
                        if lower not in self.stop_words:
                                filtered_tokens.append(token)
                return filtered_tokens

        def identify_named_entities(self, tokens: Sequence[str]) -> List[str]:
                named_entities = set(self.named_entity_terms)
                try:
                        chunked = ne_chunk(pos_tag(list(tokens)))
                        for subtree in chunked:
                                if isinstance(subtree, Tree):
                                        named_entities.add(" ".join(token for token, _ in subtree.leaves()))
                except LookupError:  # pragma: no cover - missing optional model
                        logger.warning("Named entity chunker resources are missing; returning known entities only")
                return sorted({entity for entity in named_entities if entity})

        def conditional_lemmatization(self, tokens: Sequence[str], named_entities: Sequence[str]) -> List[str]:
                named_entity_set = {entity.lower() for entity in named_entities}
                financial_terms = self.financial_terms
                lemmatized_tokens: List[str] = []
                for token, tag in pos_tag(list(tokens)):
                        lower = token.lower()
                        if lower in financial_terms or lower in named_entity_set:
                                lemmatized_tokens.append(token)
                                continue
                        if tag.startswith('V'):
                                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos='v'))
                        else:
                                lemmatized_tokens.append(self.lemmatizer.lemmatize(token))
                return lemmatized_tokens

        def extract_custom_patterns(self, text: str) -> Dict[str, List[str]]:
                extracted: Dict[str, List[str]] = {}
                for category, patterns in self._compiled_patterns.items():
                        matches: List[str] = []
                        for compiled in patterns:
                                matches.extend(compiled.findall(text))
                        if matches:
                                # Flatten tuples returned by regex groups
                                normalized_matches = [match if isinstance(match, str) else "".join(match) for match in matches]
                                extracted[category] = sorted(set(filter(None, normalized_matches)))
                return extracted

        def _remove_html_tags(self, text: str) -> str:
                return re.sub(r"<[^>]+>", " ", text)

        def _remove_punctuation(self, text: str) -> str:
                preserved_chars = "$%'%-"
                pattern = rf"[^0-9A-Za-z\s{re.escape(preserved_chars)}]"
                text = re.sub(pattern, " ", text)
                # Collapse punctuation artifacts around numbers like 1,000 or 3.5%
                text = re.sub(r"(?<=\d)[,](?=\d)", ",", text)
                return text

        def _conditional_lowercase(self, text: str) -> str:
                tokens = word_tokenize(text)
                if not tokens:
                        return text
                lowered_tokens: List[str] = []
                tagged_tokens = pos_tag(tokens)
                entity_terms = {tuple(term.lower().split()) for term in self.named_entity_terms}
                skip_indices = self._find_entity_spans([token.lower() for token in tokens], entity_terms)
                for index, (token, tag) in enumerate(tagged_tokens):
                        if index in skip_indices or tag in ("NNP", "NNPS"):
                                lowered_tokens.append(token)
                        else:
                                lowered_tokens.append(token.lower())
                return " ".join(lowered_tokens)

        def _find_entity_spans(
                self,
                lowered_tokens: Sequence[str],
                entity_terms: Iterable[Tuple[str, ...]],
        ) -> set:
                protected_indices: set = set()
                for term in entity_terms:
                        if not term:
                                continue
                        term_length = len(term)
                        for idx in range(0, len(lowered_tokens) - term_length + 1):
                                window = tuple(lowered_tokens[idx: idx + term_length])
                                if window == term:
                                        protected_indices.update(range(idx, idx + term_length))
                return protected_indices

        def _normalize_whitespace(self, text: str) -> str:
                return re.sub(r"\s+", " ", text).strip()

        def _load_stopwords(self, language: str, overrides: Iterable[str]) -> set:
                try:
                        base_stopwords = set(stopwords.words(language))
                except LookupError:  # pragma: no cover - fallback for unsupported languages
                        logger.warning("Stopword list for '%s' is unavailable; falling back to English", language)
                        base_stopwords = set(stopwords.words("english"))
                lower_overrides = {word.lower() for word in overrides}
                base_stopwords.update(lower_overrides)
                return base_stopwords

        def _extract_financial_terms(self, custom_entities: Mapping[str, Sequence[Mapping[str, Any]]]) -> set:
                financial_terms: set = set()
                for term in custom_entities.get("FINANCIAL_VOCABULARY", []):
                        value = term.get("term") if isinstance(term, Mapping) else str(term)
                        if value:
                                financial_terms.add(value.lower())
                return financial_terms

        def _collect_named_entities(self, custom_entities: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
                terms: List[str] = []
                for entity_list in custom_entities.values():
                        for entity in entity_list:
                                if isinstance(entity, Mapping) and "term" in entity:
                                        terms.append(entity["term"])
                                elif isinstance(entity, str):
                                        terms.append(entity)
                return terms

        def _build_mwe_tokenizer(
                self,
                multi_word_entries: Mapping[str, Sequence[Mapping[str, Any]]],
                named_entity_terms: Sequence[str],
        ) -> MWETokenizer:
                expressions: List[Tuple[str, ...]] = []
                for entries in multi_word_entries.values():
                        for entry in entries:
                                term = entry.get("term") if isinstance(entry, Mapping) else str(entry)
                                if term and " " in term:
                                        expressions.append(tuple(term.lower().split()))
                for term in named_entity_terms:
                        if term and " " in term:
                                expressions.append(tuple(term.lower().split()))
                if expressions:
                        return MWETokenizer(expressions)
                return MWETokenizer()

        def _compile_patterns(self, regex_patterns: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, List[re.Pattern]]:
                compiled: Dict[str, List[re.Pattern]] = {}
                for category, entries in regex_patterns.items():
                        compiled_patterns: List[re.Pattern] = []
                        for entry in entries:
                                pattern: Optional[str] = None
                                flags = 0
                                if isinstance(entry, Mapping):
                                        pattern = entry.get("pattern")
                                        for flag_name in entry.get("flags", []):
                                                flags |= getattr(re, flag_name, 0)
                                elif isinstance(entry, str):
                                        pattern = entry
                                if not pattern:
                                        continue
                                try:
                                        compiled_patterns.append(re.compile(pattern, flags))
                                except re.error as exc:  # pragma: no cover - defensive logging
                                        logger.warning("Invalid regex pattern '%s' for category '%s': %s", pattern, category, exc)
                        if compiled_patterns:
                                compiled[category] = compiled_patterns
                return compiled


def load_preprocessor_from_config(config_path: str) -> TextPreprocessor:
        """Helper that mirrors legacy module-level factory semantics."""
        return TextPreprocessor.from_config_path(config_path)


__all__ = [
        "PreprocessingConfig",
        "PreprocessingOutput",
        "TextPreprocessor",
        "load_preprocessor_from_config",
]
