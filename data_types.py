#!/usr/bin/env python3

# data_types.py

from typing import NamedTuple, Optional, Tuple

class EntityToken(NamedTuple):
	text: str
	label: str
	source: str
	start: int
	end: int
	pos_tag: Optional[str] = None
	
class ProcessedToken(NamedTuple):
	text: str
	lemma: str
	pos: str
	action: str
	source: str
	entity_type: Optional[str] = None
	
class SemanticRole(NamedTuple):
	"""Container for semantic role information"""
	predicate: str
	argument: str
	role: str
	start: int
	end: int
	
class AspectTerm(NamedTuple):
	"""Container for aspect term information"""
	text: str
	category: str
	target: Optional[str]  # The entity this aspect refers to
	position: Tuple[int, int]
	confidence: float