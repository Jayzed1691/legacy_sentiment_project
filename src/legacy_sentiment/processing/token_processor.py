#!/usr/bin/env python3

# token_processor.py

from typing import List, Dict, Any, Set, Optional, Tuple, Union
from stopword_handler import StopwordHandler
from data_types import EntityToken, ProcessedToken
import logging
import spacy

logger = logging.getLogger(__name__)

class TokenProcessor:
	"""
	Enhanced token processor with improved entity protection and POS handling.
	"""
	def __init__(self, stopword_handler: StopwordHandler):
		self.stopword_handler = stopword_handler
		
	def process_text_from_doc(self, 
							doc: spacy.tokens.Doc,
							identified_tokens: List[EntityToken]) -> List[ProcessedToken]:
		"""Process text using pre-processed spaCy Doc."""
		# Sort tokens by position for efficient lookup
		identified_tokens = sorted(identified_tokens, key=lambda x: x.start)
		
		# Create span lookup for quick entity checking
		entity_spans = self._create_entity_spans(identified_tokens)
		
		# Get stopword processing results
		basic_processed = self.stopword_handler.process_text_from_doc(doc)
		
		# Map token indices to character positions
		token_positions = {token.i: (token.idx, token.idx + len(token.text)) 
							for token in doc}
		
		return self._protect_entities_from_doc(
			doc, 
			basic_processed, 
			identified_tokens,
			entity_spans,
			token_positions
		)
	
	def _protect_entities_from_doc(self, 
							doc: spacy.tokens.Doc,
							processed_tokens: List[Tuple[str, str, str, str, str]], 
							identified_tokens: List[EntityToken],
							entity_spans: Set[int],
							token_positions: Dict[int, Tuple[int, int]]) -> List[ProcessedToken]:
		"""Protect entities from stopword removal using pre-processed Doc."""
		protected_tokens = []
		
		# Create mapping of character positions to entities
		entity_lookup = {(ent.start, ent.end): (ent.label, ent.source) 
						for ent in identified_tokens}
		
		for i, (token_text, lemma, pos, action, source) in enumerate(processed_tokens):
			if i not in token_positions:
				continue
			
			start, end = token_positions[i]
			
			# Check if token is part of an entity
			is_entity = any(start >= ent_start and end <= ent_end 
							for ent_start, ent_end in entity_lookup)
			
			if is_entity:
				# Find the containing entity
				for (ent_start, ent_end), (label, ent_source) in entity_lookup.items():
					if start >= ent_start and end <= ent_end:
						protected_tokens.append(ProcessedToken(
							text=token_text,
							lemma=lemma,
							pos=pos,
							action='Keep',
							source=f'Protected-{label}-{ent_source}',
							entity_type=label
						))
						break
			else:
				protected_tokens.append(ProcessedToken(
					text=token_text,
					lemma=lemma,
					pos=pos,
					action=action,
					source=source,
					entity_type=None
				))
				
		return protected_tokens
	
	def _protect_entities(self, 
						text: str,
						processed_tokens: List[Tuple[str, str, str, str, str]], 
						identified_tokens: List[EntityToken]) -> List[ProcessedToken]:
		"""
		Protect entities from stopword removal with enhanced linguistic information.
		
		Args:
			text (str): Original text
			processed_tokens (List[Tuple]): Basic processed tokens
			identified_tokens (List[EntityToken]): Identified entities
			
		Returns:
			List[ProcessedToken]: Protected tokens with enhanced information
		"""
		# Create span lookup for quick entity checking
		entity_spans = self._create_entity_spans(identified_tokens)
		
		# Create enhanced entity lookup with type information
		entity_texts = {
			text[token.start:token.end].lower(): {
				'label': token.label,
				'source': token.source,
				'span': (token.start, token.end),
				'pos_tag': token.pos_tag
			}
			for token in identified_tokens
		}
		
		protected_tokens = []
		current_pos = 0
		
		for token, lemma, pos, action, source in processed_tokens:
			# Find actual position of token in text
			token_start = text[current_pos:].lower().find(token.lower())
			if token_start != -1:
				token_start += current_pos
				token_end = token_start + len(token)
				
				# Determine entity type and build ProcessedToken
				entity_type = None
				final_action = action
				final_source = source
				
				# Check if token is part of any entity
				if self._is_token_in_entity(token_start, token_end, entity_spans):
					# Check if this token is a complete entity
					token_lower = token.lower()
					if token_lower in entity_texts:
						entity_info = entity_texts[token_lower]
						entity_type = entity_info['label']
						final_action = 'Keep'
						final_source = f"Protected-{entity_info['label']}-{entity_info['source']}"
						pos = entity_info['pos_tag'] or pos  # Use entity POS tag if available
					else:
						# Handle partial entity matches
						overlapping_entities = [
							ent for ent in identified_tokens
							if ent.start <= token_start < ent.end or ent.start < token_end <= ent.end
						]
						if overlapping_entities:
							entity = overlapping_entities[0]
							entity_type = entity.label
							final_action = 'Keep'
							final_source = f'Protected-Partial-{entity.label}-{entity.source}'
							pos = entity.pos_tag or pos  # Use entity POS tag if available
						else:
							final_action = 'Keep'
							final_source = 'Protected-Partial'
							
				protected_tokens.append(ProcessedToken(
					text=token,
					lemma=lemma,
					pos=pos,
					action=final_action,
					source=final_source,
					entity_type=entity_type
				))
				
				current_pos = token_end
			else:
				# Handle tokens not found in original text
				protected_tokens.append(ProcessedToken(
					text=token,
					lemma=lemma,
					pos=pos,
					action=action,
					source=f'{source}-Unmatched' if action == 'Stopword' else source,
					entity_type=None
				))
				
		return protected_tokens
	
	def _create_entity_spans(self, identified_tokens: List[EntityToken]) -> Set[int]:
		"""Create a set of all character positions covered by entities."""
		entity_spans = set()
		for token in identified_tokens:
			entity_spans.update(range(token.start, token.end))
		return entity_spans
	
	def _is_token_in_entity(self, token_start: int, token_end: int, entity_spans: Set[int]) -> bool:
		"""Check if token position overlaps with any entity span."""
		return any(pos in entity_spans for pos in range(token_start, token_end))
	
	def get_non_entity_segments(self, text: str, identified_tokens: List[EntityToken]) -> List[Tuple[int, int]]:
		"""
		Get text segments that don't contain identified entities.
		
		Args:
			text (str): Original text
			identified_tokens (List[EntityToken]): Identified entities
			
		Returns:
			List[Tuple[int, int]]: List of (start, end) positions for non-entity segments
		"""
		if not identified_tokens:
			return [(0, len(text))]
		
		segments = []
		last_end = 0
		
		for token in sorted(identified_tokens, key=lambda x: x.start):
			if token.start > last_end:
				segments.append((last_end, token.start))
			last_end = token.end
			
		if last_end < len(text):
			segments.append((last_end, len(text)))
			
		return segments