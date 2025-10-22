#!/usr/bin/env python3

"""Token processing utilities that integrate stopword handling and entity protection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import spacy

from legacy_sentiment.data_models.data_types import EntityToken, ProcessedToken
from legacy_sentiment.utils.stopword_handler import StopwordHandler

logger = logging.getLogger(__name__)


class TokenProcessor:
    """Enhanced token processor with improved entity protection and POS handling."""

    def __init__(self, stopword_handler: StopwordHandler) -> None:
        self.stopword_handler = stopword_handler

    def process_text(
        self,
        text: str,
        identified_tokens: List[EntityToken],
        *,
        nlp: Optional[spacy.language.Language] = None,
    ) -> List[ProcessedToken]:
        """Process a raw text string using an available spaCy pipeline."""
        nlp_model = nlp or getattr(self.stopword_handler, "nlp", None)
        if nlp_model is None:
            raise ValueError("A spaCy language model must be provided to process text.")

        doc = nlp_model(text)
        return self.process_text_from_doc(doc, identified_tokens)

    def process_text_from_doc(
        self,
        doc: spacy.tokens.Doc,
        identified_tokens: List[EntityToken],
    ) -> List[ProcessedToken]:
        """Process text using a pre-computed spaCy Doc."""
        ordered_tokens = sorted(identified_tokens, key=lambda token: token.start)
        entity_spans = self._create_entity_spans(ordered_tokens)
        basic_processed = self.stopword_handler.process_text_from_doc(doc)
        token_positions = {token.i: (token.idx, token.idx + len(token.text)) for token in doc}

        return self._protect_entities_from_doc(
            doc,
            basic_processed,
            ordered_tokens,
            entity_spans,
            token_positions,
        )

    def _protect_entities_from_doc(
        self,
        doc: spacy.tokens.Doc,
        processed_tokens: List[Tuple[str, str, str, str, str]],
        identified_tokens: List[EntityToken],
        entity_spans: Set[int],
        token_positions: Dict[int, Tuple[int, int]],
    ) -> List[ProcessedToken]:
        """Protect entities from stopword removal using pre-processed Doc."""
        protected_tokens: List[ProcessedToken] = []
        entity_lookup = {(ent.start, ent.end): (ent.label, ent.source) for ent in identified_tokens}

        for index, (token_text, lemma, pos, action, source) in enumerate(processed_tokens):
            if index not in token_positions:
                continue

            start, end = token_positions[index]
            is_entity = any(start >= ent_start and end <= ent_end for ent_start, ent_end in entity_lookup)

            if is_entity:
                for (ent_start, ent_end), (label, ent_source) in entity_lookup.items():
                    if start >= ent_start and end <= ent_end:
                        protected_tokens.append(
                            ProcessedToken(
                                text=token_text,
                                lemma=lemma,
                                pos=pos,
                                action="Keep",
                                source=f"Protected-{label}-{ent_source}",
                                entity_type=label,
                            )
                        )
                        break
            else:
                protected_tokens.append(
                    ProcessedToken(
                        text=token_text,
                        lemma=lemma,
                        pos=pos,
                        action=action,
                        source=source,
                        entity_type=None,
                    )
                )

        return protected_tokens

    def _protect_entities(
        self,
        text: str,
        processed_tokens: List[Tuple[str, str, str, str, str]],
        identified_tokens: List[EntityToken],
    ) -> List[ProcessedToken]:
        entity_spans = self._create_entity_spans(identified_tokens)
        entity_texts = {
            text[token.start:token.end].lower(): {
                "label": token.label,
                "source": token.source,
                "span": (token.start, token.end),
                "pos_tag": token.pos_tag,
            }
            for token in identified_tokens
        }

        protected_tokens: List[ProcessedToken] = []
        current_pos = 0

        for token, lemma, pos, action, source in processed_tokens:
            token_start = text[current_pos:].lower().find(token.lower())
            if token_start == -1:
                protected_tokens.append(
                    ProcessedToken(
                        text=token,
                        lemma=lemma,
                        pos=pos,
                        action=action,
                        source=f"{source}-Unmatched" if action == "Stopword" else source,
                        entity_type=None,
                    )
                )
                continue

            token_start += current_pos
            token_end = token_start + len(token)

            entity_type: Optional[str] = None
            final_action = action
            final_source = source

            if self._is_token_in_entity(token_start, token_end, entity_spans):
                token_lower = token.lower()
                if token_lower in entity_texts:
                    entity_info = entity_texts[token_lower]
                    entity_type = entity_info["label"]
                    final_action = "Keep"
                    final_source = f"Protected-{entity_info['label']}-{entity_info['source']}"
                    pos = entity_info["pos_tag"] or pos
                else:
                    overlapping = [
                        ent
                        for ent in identified_tokens
                        if ent.start <= token_start < ent.end or ent.start < token_end <= ent.end
                    ]
                    if overlapping:
                        entity = overlapping[0]
                        entity_type = entity.label
                        final_action = "Keep"
                        final_source = f"Protected-Partial-{entity.label}-{entity.source}"
                        pos = entity.pos_tag or pos
                    else:
                        final_action = "Keep"
                        final_source = "Protected-Partial"

            protected_tokens.append(
                ProcessedToken(
                    text=token,
                    lemma=lemma,
                    pos=pos,
                    action=final_action,
                    source=final_source,
                    entity_type=entity_type,
                )
            )

            current_pos = token_end

        return protected_tokens

    def _create_entity_spans(self, identified_tokens: List[EntityToken]) -> Set[int]:
        entity_spans: Set[int] = set()
        for token in identified_tokens:
            entity_spans.update(range(token.start, token.end))
        return entity_spans

    def _is_token_in_entity(self, token_start: int, token_end: int, entity_spans: Set[int]) -> bool:
        return any(pos in entity_spans for pos in range(token_start, token_end))

    def get_non_entity_segments(self, text: str, identified_tokens: List[EntityToken]) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        if not identified_tokens:
            return [(0, len(text))]

        sorted_tokens = sorted(identified_tokens, key=lambda token: token.start)
        last_end = 0
        for token in sorted_tokens:
            if token.start > last_end:
                segments.append((last_end, token.start))
            last_end = max(last_end, token.end)

        if last_end < len(text):
            segments.append((last_end, len(text)))

        return segments
