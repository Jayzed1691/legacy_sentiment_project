#!/usr/bin/env python3

"""Shared data models and constants used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from spacy.tokens import Token as SpacyToken


# ---------------------------------------------------------------------------
# spaCy model metadata
# ---------------------------------------------------------------------------

SPACY_MODEL_SM = "en_core_web_sm"
SPACY_MODEL_MD = "en_core_web_md"
SPACY_MODEL_LG = "en_core_web_lg"
SPACY_MODEL_TRF = "en_core_web_trf"

SPACY_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    SPACY_MODEL_SM: {
        "description": "Small English pipeline with vocabulary, syntax, entities",
        "gpu_required": False,
    },
    SPACY_MODEL_MD: {
        "description": "Medium sized English pipeline with vectors",
        "gpu_required": False,
    },
    SPACY_MODEL_LG: {
        "description": "Large English pipeline with vectors",
        "gpu_required": False,
    },
    SPACY_MODEL_TRF: {
        "description": "Transformer based English pipeline",
        "gpu_required": True,
    },
}

# Dependency labels that should expand a span when constructing matches.
SPACY_MODEL_DEPENDENCIES: Dict[str, Dict[str, Iterable[str]]] = {
    SPACY_MODEL_SM: {
        "preceding": {"amod", "compound", "det", "nummod"},
        "trailing": {"prep", "pobj", "dobj", "conj"},
    },
    SPACY_MODEL_MD: {
        "preceding": {"amod", "compound", "det", "nummod"},
        "trailing": {"prep", "pobj", "dobj", "conj"},
    },
    SPACY_MODEL_LG: {
        "preceding": {"amod", "compound", "det", "nummod"},
        "trailing": {"prep", "pobj", "dobj", "conj"},
    },
    SPACY_MODEL_TRF: {
        "preceding": {"amod", "compound", "det", "nummod"},
        "trailing": {"prep", "pobj", "dobj", "conj", "advcl"},
    },
}


# ---------------------------------------------------------------------------
# Core entity and token containers
# ---------------------------------------------------------------------------


class EntityToken(NamedTuple):
    """Span level representation of a matched entity."""

    text: str
    label: str
    source: str
    start: int
    end: int
    pos_tag: Optional[str] = None


class ProcessedToken(NamedTuple):
    """Representation of a token after stopword analysis."""

    text: str
    lemma: str
    pos: str
    action: str
    source: str
    entity_type: Optional[str] = None


@dataclass
class Token:
    """Unified token output produced by the matcher pipeline."""

    original_text: str
    start: int
    end: int
    source: str
    category: str
    label: str
    expanded_text: Optional[str] = None
    pos_tag: Optional[str] = None
    lemma: Optional[str] = None
    token: Optional[SpacyToken] = None

    def overlaps(self, start: int, end: int) -> bool:
        """Return ``True`` when the provided span overlaps with this token."""
        return not (self.end <= start or self.start >= end)

    def span(self) -> Tuple[int, int]:
        """Return the character span represented by the token."""
        return self.start, self.end


@dataclass
class SemanticRole:
    """Container for semantic role information."""

    predicate: str
    argument: str
    role: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedSemanticRole:
    """Extended semantic role enriched with dependency context."""

    predicate: str
    argument: str
    role: str
    start: int
    end: int
    token: Optional[SpacyToken] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    part_of_complex: Optional["SemanticComplex"] = None
    modifies: Optional["EnhancedSemanticRole"] = None
    modified_by: List["EnhancedSemanticRole"] = field(default_factory=list)


@dataclass
class SemanticComplex:
    """Grouping of related semantic roles discovered in a sentence."""

    base: EnhancedSemanticRole
    modifiers: List[EnhancedSemanticRole] = field(default_factory=list)
    scope: List[EnhancedSemanticRole] = field(default_factory=list)
    certainty: Optional[Dict[str, Any]] = None
    complex_type: Optional[str] = None

    def all_roles(self) -> List[EnhancedSemanticRole]:
        """Return every role associated with this complex."""
        return [self.base, *self.modifiers, *self.scope]


@dataclass
class AspectTerm:
    """Container for aspect term information."""

    text: str
    category: str
    target: Optional[str]
    position: Tuple[int, int]
    confidence: float


__all__ = [
    "EntityToken",
    "ProcessedToken",
    "Token",
    "SemanticRole",
    "EnhancedSemanticRole",
    "SemanticComplex",
    "AspectTerm",
    "SPACY_MODEL_SM",
    "SPACY_MODEL_MD",
    "SPACY_MODEL_LG",
    "SPACY_MODEL_TRF",
    "SPACY_MODEL_INFO",
    "SPACY_MODEL_DEPENDENCIES",
]