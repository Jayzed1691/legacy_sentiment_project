#!/usr/bin/env python3

# enhanced_semantic_display_handler.py

import streamlit as st
from typing import List, Dict, Any, Optional, Union
import spacy
from enhanced_semantic_role_handler import (
	EnhancedSemanticRoleHandler,
	EnhancedSemanticRole,
	SemanticComplex,
)
from semantic_role_config import SEMANTIC_ROLES

class SemanticDisplayHandler:
	"""
	Handles display of semantic analysis results.
	"""
	
	def __init__(
		self, nlp: spacy.language.Language, language_data: Optional[Dict] = None
	):
		"""Initialize with SpaCy model and optional language data."""
		self.analyzer = EnhancedSemanticRoleHandler(nlp, language_data)
		
	def analyze_text(self, text: str) -> Dict[str, Any]:
		"""Perform semantic analysis of text."""
		doc = self.analyzer.nlp(text)
		return {
			"semantic_roles": self.analyzer.extract_roles_from_doc(doc),
			"original_text": text,
		}
	
	def display_analysis(self, analysis_results: Dict[str, Any], container=st) -> None:
		"""Display semantic analysis results."""
		text = analysis_results.get("original_text", "")
		roles_and_complexes = analysis_results.get("semantic_roles",)
		
		# Display original text
		container.markdown("#### Original Text")
		container.write(text)
		
		# Display semantic complexes and roles
		container.markdown("#### Semantic Structure")
		self._display_semantic_structure(roles_and_complexes, container)
		
		# Display Prepositional Relations
		self._display_prep_relations(roles_and_complexes, container)
		
		# Display Special Roles
		self._display_special_roles(roles_and_complexes, container)
		
	def _display_semantic_structure(
		self,
		roles_and_complexes: List[Union[EnhancedSemanticRole, SemanticComplex]],
		container,
	) -> None:
		"""Display semantic structure."""
		if not roles_and_complexes:
			container.write("No semantic structure identified")
			return
		
		# Group by predicate
		predicate_structures = self._group_by_predicate(roles_and_complexes)
		
		for predicate, structure in predicate_structures.items():
			container.markdown(f"**Predicate:** {predicate}")
			self._display_predicate_structure(structure, container)
			
	def _group_by_predicate(
		self, roles_and_complexes: List[Union[EnhancedSemanticRole, SemanticComplex]]
	) -> Dict[str, List[Union[EnhancedSemanticRole, SemanticComplex]]]:
		"""Group roles and complexes by their main predicate."""
		predicate_structures: Dict[
			str, List[Union[EnhancedSemanticRole, SemanticComplex]]
		] = {}
		for item in roles_and_complexes:
			if isinstance(item, SemanticComplex):
				predicate = item.base.predicate
				if predicate not in predicate_structures:
					predicate_structures[predicate] = []  # Colon added here
				predicate_structures[predicate].append(item)
			elif isinstance(item, EnhancedSemanticRole):
				# Include all standalone roles in the predicate structure
				predicate = item.predicate
				if predicate not in predicate_structures:
					predicate_structures[predicate] = []
				predicate_structures[predicate].append(item)
		return predicate_structures
	
	def _display_predicate_structure(
		self,
		structure: List[Union[EnhancedSemanticRole, SemanticComplex]],
		container,
	) -> None:
		"""Display the structure for a single predicate."""
		for item in structure:
			if isinstance(item, SemanticComplex):
				self._display_complex(item, container)
			elif isinstance(item, EnhancedSemanticRole):
				self._display_role(item, container)
				
	def _display_complex(self, complex: SemanticComplex, container) -> None:
		"""Format and display a SemanticComplex."""
		lines = self._format_complex(complex)
		
		# Use markdown to display as a code block
		container.code("\n".join(lines))
		
	def _display_role(self, role: EnhancedSemanticRole, container) -> None:
		"""Display a standalone role."""
		container.markdown(f"- {role.argument} ({role.role})")
		
	def _format_complex(self, complex: SemanticComplex) -> List[str]:
		"""Format a semantic complex for display."""
		lines = []
		
		# Add complex type and base
		lines.append(f"COMPLEX [{complex.complex_type}]")
		lines.append(f"├── Base: {complex.base.argument} ({complex.base.role})")
		
		# Add modifiers (only if they exist)
		if complex.modifiers:
			lines.append("├── Modifiers:")
			for i, mod in enumerate(complex.modifiers):
				is_last = i == len(complex.modifiers) - 1
				prefix = "│   └── " if is_last else "│   ├── "
				lines.append(f"{prefix}{mod.argument} ({mod.role})")
				
		# Add scope (only if it exists)
		if complex.scope:
			lines.append("└── Scope:")
			for i, scope_item in enumerate(complex.scope):
				is_last = i == len(complex.scope) - 1
				prefix = "    └── " if is_last else "    ├── "
				lines.append(f"{prefix}{scope_item.argument} ({scope_item.role})")
				
		# If no modifiers or scope, indicate that
		if not complex.modifiers and not complex.scope:
			lines.append("└── No Modifiers or Scope")
			
		return lines
	
	def _display_prep_relations(
		self,
		roles_and_complexes: List[Union[EnhancedSemanticRole, SemanticComplex]],
		container,
	) -> None:
		"""Display prepositional relations analysis."""
		container.markdown("#### Prepositional Relations")
		prep_roles_dict = SEMANTIC_ROLES.get("prep_roles", {})
		
		prep_relations = []  # Colon added here
		
		for item in roles_and_complexes:
			if isinstance(item, SemanticComplex):
				# Check base role and scope for prep relations
				roles_to_check = [item.base] + item.scope
				for role in roles_to_check:
					if any(
						prep in role.role.lower() for prep in prep_roles_dict.keys()
					):
						prep_type = next(
							(
								prep_roles_dict[prep]
								for prep in prep_roles_dict
								if prep in role.role.lower()
							),
							"PREP",
						)
						prep_relations.append(
							f"• {prep_type}: {role.argument} in relation to {role.predicate}"
						)
			elif isinstance(item, EnhancedSemanticRole):
				if any(prep in item.role.lower() for prep in prep_roles_dict.keys()):
					prep_type = next(
						(
							prep_roles_dict[prep]
							for prep in prep_roles_dict
							if prep in item.role.lower()
						),
						"PREP",
					)
					prep_relations.append(
						f"• {prep_type}: {item.argument} in relation to {item.predicate}"
					)
					
		if prep_relations:
			for relation in prep_relations:
				container.write(relation)
		else:
			container.write("No prepositional relations identified")
			
	def _display_special_roles(
		self,
		roles_and_complexes: List[Union[EnhancedSemanticRole, SemanticComplex]],
		container,
	) -> None:
		"""Display special semantic roles analysis."""
		container.markdown("#### Special Semantic Roles")
		special_roles_dict = SEMANTIC_ROLES.get("special_roles", {})
		
		special_roles = []
		
		for item in roles_and_complexes:
			if isinstance(item, SemanticComplex):
				# Check base role, modifiers, and scope for special roles
				roles_to_check = [item.base] + item.modifiers + item.scope
				for role in roles_to_check:
					if role.role in special_roles_dict:
						role_description = special_roles_dict[role.role]
						special_roles.append(
							f"• {role.role}: {role.argument} ({role_description})"
						)
			elif isinstance(item, EnhancedSemanticRole):
				if item.role in special_roles_dict:
					role_description = special_roles_dict[item.role]
					special_roles.append(
						f"• {item.role}: {item.argument} ({role_description})"
					)
					
		if special_roles:
			for role in special_roles:
				container.write(role)
		else:
			container.write("No special semantic roles identified")