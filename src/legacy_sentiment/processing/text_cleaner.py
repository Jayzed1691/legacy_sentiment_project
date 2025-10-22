#!/usr/bin/env python3

"""Utility for cleaning matched text while respecting entities and MWE exceptions."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import spacy

from legacy_sentiment.data_models.data_types import EntityToken, ProcessedToken, Token
from legacy_sentiment.processing.token_processor import TokenProcessor
from legacy_sentiment.utils.stopword_handler import StopwordHandler


class TextCleaner:
    """Clean text while protecting matched entities."""

    def __init__(
        self,
        custom_named_entities: Optional[Dict[str, List[Dict[str, Any]]]],
        mwe_entries: Optional[Dict[str, List[Dict[str, Any]]]],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.custom_entities = self._extract_entities(custom_named_entities or {})
        self.exceptions = self._extract_exceptions(mwe_entries or {})
        self.config = config or {}
        self.preserve_punct = {".", "!", "?", "%"}

    def clean_text(
        self,
        text: str,
        entities: Sequence[Union[EntityToken, Tuple[str, str, str, int, int]]],
        processed_tokens: Sequence[Union[ProcessedToken, Tuple[str, str, str, str, str]]],
        use_lemma: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = True,
    ) -> str:
        protected_spans = set()
        entity_map: Dict[Tuple[int, int], str] = {}
        for entity_text, _, _, start, end in entities:
            protected_spans.update(range(start, end))
            entity_map[(start, end)] = entity_text

        cleaned_words: List[str] = []
        current_pos = 0
        last_token_was_punct = False

        for token, lemma, pos, action, source in processed_tokens:
            token_start = text[current_pos:].find(token)
            if token_start == -1:
                continue
            token_start += current_pos
            token_end = token_start + len(token)

            is_protected = any(token_start + i in protected_spans for i in range(len(token)))

            if is_protected:
                entity = next(
                    (ent_text for (start, end), ent_text in entity_map.items() if start <= token_start < end),
                    None,
                )
                if entity and (not cleaned_words or cleaned_words[-1] != entity):
                    cleaned_words.append(entity)
                    last_token_was_punct = False
            elif action == "Keep" or (action == "Stopword" and not remove_stopwords) or self._should_preserve_stopword(token):
                if self._is_punctuation(token):
                    if token in self.preserve_punct and cleaned_words and not last_token_was_punct:
                        cleaned_words[-1] = cleaned_words[-1].rstrip()
                        cleaned_words.append(token)
                        last_token_was_punct = True
                else:
                    word = lemma if use_lemma else token
                    cleaned_words.append(word.lower() if lowercase else word)
                    last_token_was_punct = False

            current_pos = token_end

        if not cleaned_words:
            return ""

        cleaned_text = cleaned_words[0]
        for word in cleaned_words[1:]:
            if not self._is_punctuation(word):
                cleaned_text += " "
            cleaned_text += word

        return cleaned_text.strip()

    def _is_punctuation(self, token: str) -> bool:
        return all(not char.isalnum() for char in token)

    def _extract_entities(self, custom_named_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Iterable[str]]:
        normalised: Dict[str, set] = {}
        for label, entries in custom_named_entities.items():
            values = normalised.setdefault(label, set())
            for entry in entries:
                if isinstance(entry, str):
                    values.add(entry.lower())
                elif isinstance(entry, dict):
                    term = entry.get("term") or entry.get("full_name")
                    if term:
                        values.add(str(term).lower())
                    for variation in entry.get("variations", []):
                        values.add(str(variation).lower())
                    for synonym in entry.get("synonyms", []):
                        values.add(str(synonym).lower())
        return normalised

    def _extract_exceptions(self, mwe_entries: Dict[str, List[Dict[str, Any]]]) -> set:
        exceptions: set = set()
        for entries in mwe_entries.values():
            for entry in entries:
                if isinstance(entry, str):
                    exceptions.update(entry.lower().split())
                elif isinstance(entry, dict):
                    term = entry.get("term")
                    if term:
                        exceptions.update(str(term).lower().split())
                    for synonym in entry.get("synonyms", []):
                        exceptions.update(str(synonym).lower().split())
        return exceptions

    def _should_preserve_stopword(self, token: str) -> bool:
        return token.lower() in self.exceptions


def _to_entity_token(match: Union[Token, EntityToken, Tuple[Any, ...]]) -> EntityToken:
    if isinstance(match, EntityToken):
        return match
    if isinstance(match, Token):
        return EntityToken(match.original_text, match.label, match.source, match.start, match.end, match.pos_tag)
    if isinstance(match, tuple) and len(match) >= 5:
        return EntityToken(str(match[0]), str(match[1]), str(match[2]), int(match[3]), int(match[4]))
    raise TypeError(f"Unsupported match type for conversion: {type(match)!r}")


def clean_matched_text(
    text: str,
    matches: Optional[Sequence[Union[Token, EntityToken, Tuple[Any, ...]]]] = None,
    *,
    use_lemma: bool = True,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    stopword_handler: Optional[StopwordHandler] = None,
    nlp: Optional[spacy.language.Language] = None,
) -> str:
    if not text:
        return ""

    handler = stopword_handler or StopwordHandler([])
    nlp_model = nlp or handler.nlp

    entity_tokens: List[EntityToken] = []
    if matches:
        for match in matches:
            try:
                entity_tokens.append(_to_entity_token(match))
            except TypeError:
                continue

    processor = TokenProcessor(handler)
    doc = nlp_model(text)
    processed_tokens = processor.process_text_from_doc(doc, entity_tokens)

    cleaner = TextCleaner({}, {})
    return cleaner.clean_text(
        text=text,
        entities=entity_tokens,
        processed_tokens=processed_tokens,
        use_lemma=use_lemma,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
    )
