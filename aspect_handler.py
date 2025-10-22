#!/usr/bin/env python3

# aspect_handler.py

import spacy
from typing import List, Dict, Any, Tuple, Optional, NamedTuple, Set
from data_types import SemanticRole, AspectTerm
import logging
from custom_file_utils import load_json_file

logger = logging.getLogger(__name__)

class AspectHandler:
	""" Handles identification and classification of entity aspects. """
	
	def __init__(self, 
					nlp: spacy.language.Language,
					aspect_config_file: Optional[str] = None):
			self.nlp = nlp
		
			# Load aspect categories from config file
			if aspect_config_file:
					self.ASPECT_CATEGORIES = load_json_file(aspect_config_file)
					logger.info(f"AspectHandler loaded categories: {list(self.ASPECT_CATEGORIES.keys())}")
				
					# Pre-process term lemmas during initialization
					self.term_lemmas = {}
					for category, subcats in self.ASPECT_CATEGORIES.items():
							self.term_lemmas[category] = {}
							for subcat, terms in subcats.items():
									self.term_lemmas[category][subcat] = {
									term.lower(): self.nlp(term)[0].lemma_.lower() 
										for term in terms
									} 
					logger.info("Pre-processed term lemmas for aspect matching")
				
			else:
					logger.warning("No aspect configuration file provided, using default categories")
					self.ASPECT_CATEGORIES = {}
					self.term_lemmas = {}
				
			# Aspect-indicating dependency patterns
			self.ASPECT_DEPS = {
					'nsubj': 0.8,  # Subject of aspect
					'dobj': 0.8,   # Direct object
					'pobj': 0.7,   # Object of preposition
					'amod': 0.9,   # Adjectival modifier
					'compound': 0.7 # Compound noun
			}
		
	def extract_aspects_from_doc(self,
					doc: spacy.tokens.Doc,
					noun_chunks: List[Any],
					semantic_roles: List[SemanticRole]) -> List[AspectTerm]:
			"""Extract aspects using pre-processed Doc."""
			aspects = []
		
			# Track processed spans to avoid duplicates
			processed_spans = set()
		
			# Process noun chunks first
			aspects.extend(self._process_noun_chunks(doc, noun_chunks, processed_spans))
		
			# Process semantic roles for additional aspects
			aspects.extend(self._process_semantic_roles(doc, semantic_roles, processed_spans))
		
			# Merge and deduplicate aspects
			return self._merge_aspects(doc.text, aspects)
	
	def _process_noun_chunks(self, 
			doc: spacy.tokens.Doc,
			noun_chunks: List[Any],
			processed_spans: Set[Tuple[int, int]]) -> List[AspectTerm]:
		"""Process noun chunks to identify aspects."""
		aspects = []
		text = doc.text
		
		for chunk in noun_chunks:
			chunk_start = chunk.start
			chunk_end = chunk.end
			
			if (chunk_start, chunk_end) in processed_spans:
				continue
			
			doc_span = doc.char_span(chunk_start, chunk_end)
			if not doc_span:
				continue
			
			# Check the full chunk text first for compound matches
			full_text = doc_span.text.lower()
			for category, subcats in self.ASPECT_CATEGORIES.items():
				for subcat, terms in subcats.items():
					if any(term.lower() in full_text for term in terms):
						confidence = 0.9  # Higher confidence for full matches
						target = self._find_aspect_target(doc_span)
						
						aspects.append(AspectTerm(
							text=chunk.text,
							category=category,
							target=target,
							position=(chunk_start, chunk_end),
							confidence=confidence
						))
						processed_spans.add((chunk_start, chunk_end))
						break
				if (chunk_start, chunk_end) in processed_spans:
					break
				
			# If no full match, try individual tokens
			if (chunk_start, chunk_end) not in processed_spans:
				category, confidence = self._categorize_span(doc_span)
				if category and confidence > 0.5:
					target = self._find_aspect_target(doc_span)
					aspects.append(AspectTerm(
						text=chunk.text,
						category=category,
						target=target,
						position=(chunk_start, chunk_end),
						confidence=confidence
					))
					processed_spans.add((chunk_start, chunk_end))
					
		return aspects
	
	def _process_semantic_roles(self,
							doc: spacy.tokens.Doc,
							semantic_roles: List[SemanticRole],
							processed_spans: Set[Tuple[int, int]]) -> List[AspectTerm]:
		"""Process semantic roles to identify aspects."""
		aspects = []
		text = doc.text
		
		for role in semantic_roles:
			# Skip if span already processed
			if (role.start, role.end) in processed_spans:
				continue
			
			# Focus on arguments that might contain aspects
			if role.role in {'PATIENT', 'INSTRUMENT', 'ATTRIBUTE'}:
				span = doc.char_span(role.start, role.end)
				if not span:
					continue
				
				category, confidence = self._categorize_span(span)
				if category and confidence > 0.5:
					# Use predicate as target for aspects in arguments
					aspects.append(AspectTerm(
						text=span.text,
						category=category,
						target=role.predicate,
						position=(role.start, role.end),
						confidence=confidence
					))
					processed_spans.add((role.start, role.end))
					
		return aspects
	
	def _categorize_span(self, 
			span: spacy.tokens.Span) -> Tuple[Optional[str], float]:
			"""
			Categorize a text span into aspect categories.
			Returns category and confidence score.
			"""
			max_score = 0
			best_category = None
		
			# Use span's parent doc for lemmatization
			doc = span.doc
		
			# Check each token in span
			for token in span:
					# Ensure token is properly part of doc
					if not token.has_vector or not token.is_alpha:
							continue
				
					token_lower = token.lemma_.lower()
					token_text = token.text.lower()
				
					# Debug log
					logger.debug(f"Checking token: {token_text} (lemma: {token_lower})")
				
					# Check each category and its subcategories
					for category, subcats in self.ASPECT_CATEGORIES.items():
							score = 0
							for subcat, terms in subcats.items():
									# Check both exact matches and stem/lemma variations
									term_matches = [term.lower() for term in terms]
									lemma_matches = [
											self.term_lemmas[category][subcat][term]
											for term in terms
									] 
								
									if (token_lower in lemma_matches or 
											token_text in term_matches or
											any(token_lower.startswith(term.lower()) for term in terms)):
								
											logger.debug(f"Found match: {token_text} in {category}.{subcat}")
											# Base score from term match
											score = 0.7
											# Boost score based on dependencies
											if token.dep_ in self.ASPECT_DEPS:
													score += self.ASPECT_DEPS[token.dep_]
													logger.debug(f"Boosted score for dependency {token.dep_}")
											score = min(score, 1.0)
								
											if score > max_score:
													max_score = score
													best_category = category
													break
								
			logger.debug(f"Final category: {best_category}, score: {max_score}")
			return best_category, max_score
	
	def _find_aspect_target(self, span: spacy.tokens.Span) -> Optional[str]:
		"""Find the entity or concept that an aspect refers to."""
		for token in span:
			# Check for entity mentions
			if token.ent_type:
				return token.text
			
			# Check for compound nouns that might be targets
			if token.dep_ == "compound" and token.head.pos_ == "NOUN":
				return token.head.text
			
		return None
	
	def _merge_aspects(self, text: str, aspects: List[AspectTerm]) -> List[AspectTerm]:
		"""Merge overlapping or related aspects."""
		if not aspects:
			return []
		
		# Sort by position and confidence
		sorted_aspects = sorted(aspects, 
							key=lambda x: (x.position[0], -x.confidence))
		
		merged = []
		current = sorted_aspects[0]
		
		for next_aspect in sorted_aspects[1:]:
			# Check for overlap or adjacency
			if (current.position[1] >= next_aspect.position[0] and
				current.category == next_aspect.category):
				# Merge if same category and overlapping
				current = AspectTerm(
					text=text[current.position[0]:max(current.position[1], 
													next_aspect.position[1])],
					category=current.category,
					target=current.target or next_aspect.target,
					position=(current.position[0], 
							max(current.position[1], next_aspect.position[1])),
					confidence=max(current.confidence, next_aspect.confidence)
				)
			else:
				merged.append(current)
				current = next_aspect
				
		merged.append(current)
		return merged