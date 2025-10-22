#!/usr/bin/env python3

# transcript_structures.py

from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict

from legacy_sentiment.data_models.data_types import SemanticRole

@dataclass
class DialogueEntry:
	speaker: str
	role: str
	text: str
	semantic_roles: List[SemanticRole] = field(default_factory=list)
	
@dataclass
class Section:
	name: str
	dialogues: List[DialogueEntry] = field(default_factory=list)
	subsections: List['Section'] = field(default_factory=list)
	
@dataclass
class TranscriptData:
	sections: List[Section] = field(default_factory=list)
	speakers: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))