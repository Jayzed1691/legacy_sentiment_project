#!/usr/bin/env python3

# semantic_role_handler.py

import spacy
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
import logging
from data_types import SemanticRole

logger = logging.getLogger(__name__)

class SemanticRoleHandler:
	"""Handles extraction of semantic roles from text using dependency parsing."""
	
	def __init__(self, nlp: spacy.language.Language):
		self.nlp = nlp
		
		# Core semantic roles we'll identify
		self.CORE_ROLES = {
			'nsubj': 'AGENT',      # The doer of the action
			'dobj': 'PATIENT',     # The receiver of the action
			'iobj': 'RECIPIENT',   # The recipient of the transfer
			'pobj': 'INSTRUMENT',  # The means by which action is performed
		}
		
		# Extended roles from prepositional and adverbial modifiers
		self.PREP_ROLES = {
			'in': 'LOCATION',
			'at': 'LOCATION',
			'on': 'LOCATION',
			'to': 'DESTINATION',
			'from': 'SOURCE',
			'with': 'INSTRUMENT',
			'by': 'AGENT',
			'for': 'PURPOSE',
			'during': 'TEMPORAL',
			'before': 'TEMPORAL',
			'after': 'TEMPORAL'
		}
		
	def extract_roles_from_doc(self, doc: spacy.tokens.Doc) -> List[SemanticRole]:
		"""Extract semantic roles from pre-processed Doc."""
		roles = []
		
		for token in doc:
			# Focus on verbs as predicates
			if token.pos_ == "VERB":
				# Get core arguments
				roles.extend(self._get_core_arguments(token))
				# Get prepositional arguments
				roles.extend(self._get_prep_arguments(token))
				# Get temporal and locative modifiers
				roles.extend(self._get_modifiers(token))
				
		return roles
	
	def get_predicate_argument_structure(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
		"""Original method - get predicate-argument structure from text string."""
		doc = self.nlp(text)
		return self.get_predicate_argument_structure_from_doc(doc)
	
	def get_predicate_argument_structure_from_doc(self, doc: spacy.tokens.Doc) -> Dict[str, List[Dict[str, Any]]]:
		"""Get structured predicate-argument relationships from pre-processed Doc."""
		roles = self.extract_roles_from_doc(doc)
		
		# Group by predicate
		pred_args = {}
		for role in roles:
			if role.predicate not in pred_args:
				pred_args[role.predicate] = []
			pred_args[role.predicate].append({
				'argument': role.argument,
				'role': role.role,
				'position': (role.start, role.end)
			})
			
		return pred_args
	
	def _get_core_arguments(self, verb: spacy.tokens.Token) -> List[SemanticRole]:
		"""Extract core arguments (subject, object) for a verb."""
		roles = []
		doc = verb.doc
		
		for child in verb.children:
			if child.dep_ in self.CORE_ROLES:
				# Create span for argument using doc
				argument_tokens = list(child.subtree)
				if argument_tokens:
					start_token = argument_tokens[0]
					end_token = argument_tokens[-1]
					span = doc[start_token.i:end_token.i + 1]
					
					roles.append(SemanticRole(
						predicate=verb.text,
						argument=span.text,
						role=self.CORE_ROLES[child.dep_],
						start=span.start_char,
						end=span.end_char
					))
		return roles
	
	def _get_prep_arguments(self, verb: spacy.tokens.Token) -> List[SemanticRole]:
		"""Extract prepositional arguments."""
		roles = []
		
		for prep in verb.children:
			if prep.dep_ == 'prep' and prep.text in self.PREP_ROLES:
				for pobj in prep.children:
					if pobj.dep_ == 'pobj':
						# Get the full noun phrase
						argument = ' '.join([w.text for w in pobj.subtree])
						roles.append(SemanticRole(
							predicate=verb.text,
							argument=argument,
							role=self.PREP_ROLES[prep.text],
							start=pobj.idx,
							end=pobj.idx + len(argument)
						))
						
		return roles
	
	def _get_modifiers(self, verb: spacy.tokens.Token) -> List[SemanticRole]:
		"""Extract temporal and locative modifiers."""
		roles = []
		
		for child in verb.children:
			if child.dep_ == 'advmod' and child.pos_ == 'ADV':
				# Temporal adverbs
				if any(temp in child.text.lower() 
						for temp in ['today', 'yesterday', 'tomorrow', 'now']):
					roles.append(SemanticRole(
						predicate=verb.text,
						argument=child.text,
						role='TEMPORAL',
						start=child.idx,
						end=child.idx + len(child.text)
					))
			elif child.dep_ == 'npadvmod' and child.ent_type_ in ['DATE', 'TIME']:
				# Date/time entities
				roles.append(SemanticRole(
					predicate=verb.text,
					argument=child.text,
					role='TEMPORAL',
					start=child.idx,
					end=child.idx + len(child.text)
				))
				
		return roles