#!/usr/bin/env python3

# enhanced_semantic_role_handler.py

import logging
from typing import Any, Dict, List, Optional, Union

import spacy
from spacy.tokens import Doc

from legacy_sentiment.config.semantic_role_config import SEMANTIC_ROLES
from legacy_sentiment.data_models.data_types import (
        EnhancedSemanticRole,
        SemanticComplex,
        Token,
)
try:
        from legacy_sentiment.processing.unified_matcher_refactored import (
                get_excluded_positions,
                is_position_excluded,
        )
except ImportError:  # pragma: no cover - legacy fallback
        from superceded.unified_matcher_refactored import (  # type: ignore
                get_excluded_positions,
                is_position_excluded,
        )

# Configure the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnhancedSemanticRoleHandler:
	"""
	Handles extraction of enhanced semantic roles and complex semantic structures from text.
	"""
	
	def __init__(
		self,
		nlp: spacy.language.Language,
		semantic_role_data: Optional[Dict] = None,
	):
		"""
		Initializes the handler with a spaCy model and semantic role data.

		Args:
			nlp: Loaded spaCy language model.
			semantic_role_data: Dictionary containing domain-specific semantic role mappings.
		"""
		self.nlp = nlp
		self.semantic_role_data = semantic_role_data or {}
		
	def _is_modifiable_state(self, token: spacy.tokens.Token) -> bool:
		"""
		Check if a token represents a state that can be modified.
		For now, we'll consider verbs, adjectives, and adverbs as modifiable.
		"""
		return token.pos_ in {"VERB", "ADJ", "ADV"}
	
	def _get_modifying_tokens(self, token: spacy.tokens.Token) -> List[spacy.tokens.Token]:
		"""
		Get tokens that modify the given token based on dependency labels.
		This is a simplified version for now.
		"""
		modifiers = []
		for child in token.children:
			if child.dep_ in {"amod", "advmod", "prep", "agent"}:  # Common modifiers
				modifiers.append(child)
		return modifiers
	
	def _determine_role(self, token: spacy.tokens.Token, match_info: Optional[Token] = None) -> str:
		"""
		Determine the semantic role of a token, considering context.
		"""
		dep_lower = token.dep_.lower()
		head_dep_lower = token.head.dep_.lower()
		
		if match_info and match_info.label in {"FINANCIAL_TERMS", "PERCENT"}:
			return "ENTITY"
		
		# Check for core arguments based on dependency
		if dep_lower == "nsubj":
			return "AGENT"
		elif dep_lower in {"dobj", "iobj", "pobj", "attr"}:
			return "PATIENT"
		elif dep_lower == "prep":
			return "PREPOSITION"
		elif dep_lower == "nsubjpass":
			return "THEME"
		elif dep_lower == "agent":
			return "AGENT"
		elif dep_lower == "expl":
			return "EXPLETIVE"
		
		# Check for specific patterns or POS tags
		if token.pos_ in {"VERB", "AUX"}:
			return "ACTION"
		elif token.pos_ == "ADJ":
			return "MODIFIER"
		elif token.pos_ == "ADV":
			return "MANNER"
		elif token.pos_ == "NOUN":
			if head_dep_lower == "compound":
				return "ENTITY"  # If part of a compound noun, consider it part of a larger entity
			return "ENTITY"  # A general category for nouns
		
		
		
		# 1. Parent and Grandparent Node Analysis
		parent = token.head
		grandparent = parent.head if parent.head!= parent else None
		
		if parent:
			# Example: If parent is a verb with a 'give' lemma and token is a direct object, assign 'RECIPIENT' role
			if parent.pos_ == "VERB" and parent.lemma_ == "give" and token.dep_ == "dobj":
				return "RECIPIENT"
			# Add more rules based on parent's POS, dependency, and lemma
			elif parent.pos_ == "ADJ" and token.dep_ == "prep":
				return "ATTRIBUTE"  # Example: Assign "ATTRIBUTE" role to objects of prepositions modifying adjectives
			
		if grandparent:
			# Example: If grandparent is a verb and parent is a preposition, analyze prepositional phrase
			if grandparent.pos_ == "VERB" and parent.dep_ == "prep":
				# Analyze the preposition and its object to assign roles
				if token.dep_ == "pobj" and token.pos_ == "NOUN":
					return "LOCATION"  # Example: Assign "LOCATION" role to noun objects of prepositions with verb grandparent
			# Add more rules based on grandparent's POS, dependency, and lemma
				
		# 2. Word Lists and Patterns
		text = token.text.lower()
		lemma = token.lemma_.lower()
		
		# Utilize semantic_role_data.json
		for category, roles in self.semantic_role_data.items():
			if text in roles:
				return roles[text]  # Assign role directly from the data
			
		# Utilize custom_language_data.json
		for category, patterns in self.custom_language_data.items():
			for pattern in patterns:
				if lemma == pattern["lemma"]:  # Match based on lemma
					return category  # Assign role based on category in custom_language_data.json
			
		# Example: Use custom word lists or patterns
		if lemma in ["increase", "rise", "grow"]:
			return "INCREASE"
		elif lemma in ["decrease", "fall", "decline"]:
			return "DECREASE"
		elif token.pos_ == "ADP" and token.head.pos_ == "VERB":
			return "DIRECTION"  # Example: Assign "DIRECTION" role to prepositions modifying verbs
		
		# Fallback to more general roles based on dependency
		if dep_lower == "root":
			return "ACTION"
		elif dep_lower == "advmod":
			return "MANNER"
		
		# Default fallback
		return "NONE"
	
	def _create_enhanced_role(
		self, token: spacy.tokens.Token, match_info: Optional[Token] = None
	) -> EnhancedSemanticRole:
		"""Create an EnhancedSemanticRole object."""
		return EnhancedSemanticRole(
			predicate=token.head.text if token.dep_!= "ROOT" else token.text,
			argument=token.text,
			role=self._determine_role(token, match_info),
			start=token.idx,
			end=token.idx + len(token.text),
			token=token,
			# Placeholder for complex relationships
			part_of_complex=None,
			modifies=None,
			modified_by= [],
		)
	
	def analyze_semantic_complex(
		self, token: spacy.tokens.Token, match_info: Optional[Token] = None
	) -> Optional[SemanticComplex]:
		"""
		Analyze a token and create a SemanticComplex if applicable.
		"""
		if not self._is_modifiable_state(token):
			return None
		
		modifiers = self._get_modifying_tokens(token)
		
		# Create roles
		base_role = self._create_enhanced_role(token, match_info)
		
		if match_info:
			base_role.argument = match_info.original_text
			base_role.start = match_info.start
			base_role.end = match_info.end
			
		modifier_roles = []
		for mod in modifiers:
			modifier_roles.append(self._create_enhanced_role(mod, match_info))
			
		# Identify core arguments (subjects, objects) based on dependencies
		core_arguments = []
		for child in token.children:
			if child.dep_ in {"nsubj", "dobj", "iobj", "pobj", "nsubjpass", "agent", "attr"}:
				core_arguments.append(self._create_enhanced_role(child, match_info))
				
		# Get scope
		scope_roles = self._determine_scope(token, modifiers, match_info)
		
		complex = SemanticComplex(
			base=base_role,
			modifiers=modifier_roles,
			scope=scope_roles,
			certainty=None,  # Removed certainty analysis
			complex_type="STATEMENT",  # Default type
		)
		
		# Ensure complex is created even if no modifiers or scope
		if not modifier_roles:
			complex.modifiers = []
		if not scope_roles:
			complex.scope = []
			
		# Update relationships
		base_role.part_of_complex = complex
		for arg in core_arguments:
			arg.part_of_complex = complex
			arg.modifies = base_role
			base_role.modified_by.append(arg)
		for modifier in modifier_roles:
			modifier.part_of_complex = complex
			modifier.modifies = base_role
			base_role.modified_by.append(modifier)
		for scope in scope_roles:
			if scope and scope not in modifier_roles and scope not in core_arguments and scope!= base_role:
				scope.part_of_complex = complex
				scope.modifies = base_role
				base_role.modified_by.append(scope)
				
		return complex
	
	def extract_roles_from_doc(
		self, doc: spacy.tokens.Doc, matches: Optional[List[Token]] = None
	) -> List[Union[EnhancedSemanticRole, SemanticComplex]]:
		"""
		Extract semantic roles and complexes from a spaCy Doc, utilizing identified token matches.
		This is the main entry point for role extraction.
		"""
		roles_and_complexes: List[Union[EnhancedSemanticRole, SemanticComplex]] = []
		
		# Handle negation using dependencies
		negation_roles = self.handle_negation_with_dependencies(doc)
		roles_and_complexes.extend(negation_roles)
		
		# Sort matches by start index, longest first
		if matches:
			matches = sorted(matches, key=lambda match: (match.start, -match.end), reverse=False)
			
		excluded_positions = set()
		if matches:
			for match in matches:
				if not is_position_excluded(match.start, match.end, excluded_positions):
					# Use the existing token from the match
					token = match.token
					if token:
						# Attempt to create a complex from the matched tokens
						complex = self.analyze_semantic_complex(token, match)  # Use the root as the base for the complex
						if complex:
							roles_and_complexes.append(complex)
						else:
							# If not part of a complex, create a standalone role
							if token.dep_.lower() in {'root', 'nsubj', 'dobj', 'iobj', 'pobj'}:
								role = self._create_enhanced_role(token, match)
								role.argument = match.original_text
								role.start = match.start
								role.end = match.end
								roles_and_complexes.append(role)
						excluded_positions.update(range(match.start, match.end))
						
		return roles_and_complexes
	
	def handle_negation_with_dependencies(self, doc: Doc) -> List[EnhancedSemanticRole]:
		"""
		Handles negation using dependency parsing and creates EnhancedSemanticRole objects.
		"""
		negated_roles = []
		for token in doc:
			if token.dep_ == "neg":
				head = token.head
				negated_role = EnhancedSemanticRole(
					predicate=head.text,
					argument=head.text,
					role="NEGATED_HEAD",  # Or a more specific role if needed
					start=head.idx,
					end=head.idx + len(head.text),
					metadata={"negation": "negated by " + token.text},
					token=head,
				)
				negated_roles.append(negated_role)
		return negated_roles
	
	def _determine_scope(self, token: spacy.tokens.Token, modifiers: List[spacy.tokens.Token], match_info: Optional[Token] = None) -> List[EnhancedSemanticRole]:
		"""
		Determines the scope of a complex.
		"""
		scope = []
		
		for child in token.children:
			if child not in modifiers and child.dep_ not in {"nsubj", "dobj", "iobj", "pobj", "nsubjpass", "agent", "attr"}:
				scope.append(self._create_enhanced_role(child, match_info))
				
		return scope