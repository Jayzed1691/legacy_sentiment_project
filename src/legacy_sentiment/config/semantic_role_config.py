#!/usr/bin/env python3

# semantic_role_config.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional


# Update existing CertaintyType and CertaintyDegree
class CertaintyType(Enum):
	"""Enhanced type of certainty expression."""
	POSITIVE = "POSITIVE"  # expressions of certainty
	NEGATIVE = "NEGATIVE"  # expressions of uncertainty
	NEUTRAL = "NEUTRAL"    # no clear certainty indication
	HEDGED = "HEDGED"      # hedged or qualified certainty
	MODAL = "MODAL"        # modal qualification
	NEGATED = "NEGATED"    # negated certainty
	
class CertaintyDegree(Enum):
	"""Degree of certainty."""
	ABSOLUTE = "ABSOLUTE"
	HIGH = "HIGH"
	MODERATE = "MODERATE"
	LOW = "LOW"
	UNCERTAIN = "UNCERTAIN"
	
class CertaintyAssessment:
	"""Represents a certainty assessment for a semantic role or complex."""
	
	def __init__(
		self,
		type: CertaintyType,
		degree: CertaintyDegree,
		modifiers: Optional[List[str]] = None,
		scope: Optional[str] = None
	):
		self.type = type
		self.degree = degree
		self.modifiers = modifiers or []
		self.scope = scope or ""
		
	def __str__(self):
		return f"{self.type.value} ({self.degree.value}) - Modifiers: {self.modifiers}, Scope: '{self.scope}'"
	
	
# Semantic role types including certainty-related roles
SEMANTIC_ROLES: Dict[str, Dict[str, str]] = {
	"core_roles": {
		# Core arguments
		"nsubj": "AGENT",  # Nominal subject
		"dobj": "PATIENT",  # Direct object
		"iobj": "RECIPIENT",  # Indirect object
		"pobj": "INSTRUMENT",  # Object of preposition
		"csubj": "CAUSE",  # Clausal subject
		"ccomp": "CONTENT",  # Clausal complement
		"xcomp": "RESULT",  # Open clausal complement
		# Additional core dependencies
		"nsubjpass": "THEME",  # Passive nominal subject
		"agent": "AGENT",  # Agent in passive construction
		"expl": "EXPLETIVE",  # Expletive
		"attr": "ATTRIBUTE",  # Attribute
		"ROOT": "ACTION",  # Root (main predicate)
		# Extended core roles
		"nmod": "MODIFIER",  # Nominal modifier
		"amod": "MODIFIER",  # Adjectival modifier
		"advmod": "MANNER",  # Adverbial modifier
		"aux": "AUXILIARY",  # Auxiliary
		"auxpass": "AUXILIARY",  # Passive auxiliary
	},
	"prep_roles": {
		"in": "LOCATION",
		"at": "LOCATION",
		"on": "LOCATION",
		"to": "DESTINATION",
		"from": "SOURCE",
		"with": "INSTRUMENT",
		"by": "AGENT",
		"for": "PURPOSE",
		"during": "TEMPORAL",
		"before": "TEMPORAL",
		"after": "TEMPORAL",
		"through": "PATH",
		"via": "PATH",
		"about": "TOPIC",
		"of": "POSSESSOR",
	},
	"special_roles": {
		"EXPERIENCER": "Entity experiencing a state",
		"STIMULUS": "Entity causing an experience",
		"FORCE": "Non-agent cause",
		"BENEFICIARY": "Entity benefiting from action",
		"COMPARISON": "Entity being compared to",
		"PREDICTOR": "Entity making prediction",
		"CAUSE": "Reason for occurrence",
	},
	"certainty_roles": {
		"BELIEF": "Entity expressing belief or confidence",
		"POSSIBILITY": "Expression of possibility",
		"PREDICTION": "Future-oriented statement",
		"EVIDENCE": "Supporting evidence or basis",
		"PROBABILITY": "Expression of likelihood",
		"NEGATOR": "Negates or inverts meaning",
		"INTENSIFIER": "Strengthens or weakens expression",
		"CERTAINTY_MARKER": "Explicitly marks certainty level",
		"UNCERTAINTY_MARKER": "Explicitly marks uncertainty",
	},
}

# Dependency modifiers including certainty-related modifiers
DEPENDENCY_MODIFIERS: Dict[str, str] = {
	# Core syntactic dependencies
	"amod": "attributive",
	"compound": "compound",
	"nummod": "numeric",
	"npadvmod": "adverbial",
	"advmod": "adverbial",
	"nmod": "nominal",
	"quantmod": "quantitative",
	"prep": "prepositional",
	# Additional dependencies
	"acl": "clause",
	"appos": "appositive",
	"aux": "auxiliary",
	"case": "case_marking",
	"cc": "coordination",
	"det": "determiner",
	"mark": "marker",
	"neg": "negation",
	"poss": "possessive",
}

# Temporal markers
TEMPORAL_MARKERS: Dict[str, Set[str]] = {
	"exact_time": {
		"today",
		"yesterday",
		"tomorrow",
		"now",
		"currently",
		"presently",
		"immediately",
		"instantly",
		"promptly",
	},
	"relative_time": {
		"previously",
		"recently",
		"formerly",
		"earlier",
		"later",
		"soon",
		"eventually",
		"subsequently",
		"thereafter",
	},
	"fiscal_periods": {
		"quarter",
		"year",
		"month",
		"week",
		"fiscal",
		"annual",
		"quarterly",
		"monthly",
		"ytd",
		"q1",
		"q2",
		"q3",
		"q4",
	},
	"time_relations": {
		"before",
		"after",
		"during",
		"within",
		"throughout",
		"since",
		"until",
		"following",
		"preceding",
		"concurrent",
		"simultaneous",
	},
}

# Add new constants
CERTAINTY_MARKERS = {
	"negation": {
		"basic": {"not", "no", "never", "neither", "nor"},
		"compound": {"rule out", "eliminate", "exclude", "without", "lack"}
	},
	"modal": {
		"possibility": {"can", "could", "may", "might", "possibly", "potentially", "likely"},
		"necessity": {"must", "should", "have to", "need to", "required"},
		"prediction": {"will", "would", "going to", "expect", "anticipate", "forecast"}
	},
	"hedge": {
		"approximators": {"about", "around", "approximately", "nearly", "almost", "roughly"},
		"qualifiers": {"somewhat", "quite", "rather", "fairly", "relatively", "kind of", "sort of"},
		"limiters": {"mainly", "mostly", "generally", "typically", "largely"},
		"downtoners": {"slightly", "a bit", "some", "few", "a little"}
	},
	"intensifier": {
		"strong": { "very", "really", "highly", "extremely", "significantly", "substantially", "considerably", "greatly", "tremendously"},
		"weak": {"somewhat", "fairly", "slightly"}
	},
	"certainty_phrase": {
		"definite": {"certainly", "definitely", "absolutely", "undoubtedly"},
		"probable": {"probably", "likely", "almost certainly"},
		"possible": {"possibly", "potentially", "maybe"}
	}
}

# Business role indicators
BUSINESS_ROLE_INDICATORS: Dict[str, str] = {
	"revenue": "THEME",
	"profit": "THEME",
	"cost": "THEME",
	"margin": "THEME",
	"customer": "BENEFICIARY",
	"competitor": "FORCE",
	"market": "LOCATION",
	"guidance": "PREDICTOR",
	"forecast": "PREDICTOR",
	"impact": "FORCE",
	"growth": "THEME",
	"performance": "THEME",
	"strategy": "THEME",
	"product": "ENTITY",
	"service": "ENTITY",
	"investment": "ACTION",
	"acquisition": "ACTION",
	"merger": "ACTION",
	"divestiture": "ACTION",
	"buyback": "ACTION",
	"innovation": "THEME",
	"demand": "THEME",
	"supply": "THEME",
	"price": "ATTRIBUTE",
	"volume": "METRIC"
}

# Complex types for semantic analysis
COMPLEX_TYPES: Dict[str, str] = {
	"STATEMENT": "Basic statement or assertion",
	"BELIEF": "Expression of belief or opinion",
	"PREDICTION": "Future-oriented statement",
	"ASSESSMENT": "Evaluation or judgment",
	"POSSIBILITY": "Expression of possibility",
	"EVIDENCE": "Factual or evidential statement",
}