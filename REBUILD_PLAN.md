# Legacy Sentiment Project - Comprehensive Rebuild Plan

## Executive Summary

This document outlines a systematic approach to rebuilding the legacy_sentiment_project, a transcript analysis system focused on financial earnings calls. The project successfully implemented preprocessing, entity recognition, and display capabilities, but sophisticated linguistic analysis components underdelivered and require reevaluation.

---

## Current State Assessment

### ✅ Working Components (Preserve & Test)

1. **Transcript Ingestion** (src/legacy_sentiment/ingestion/)
   - JSON parser with nested section support
   - TXT parser with section/subsection markers
   - PDF parser (stub, needs completion if required)
   - TranscriptHandler orchestrator

2. **Preprocessing Pipeline** (src/legacy_sentiment/processing/preprocessing.py)
   - Text cleaning (HTML, whitespace, punctuation)
   - NLTK-based tokenization with MWE awareness
   - Stopword removal with financial term preservation
   - Conditional lemmatization (protects entities/financial terms)
   - Regex pattern extraction
   - **Status: Core functionality working as intended**

3. **Entity Recognition** (Multiple Handlers)
   - Custom entity handler: Dictionary-based matching with variations
   - MWE handler: Multi-word expression detection with Aho-Corasick
   - Regex handler: Pattern-based extraction (dates, currency, percentages, time periods)
   - spaCy NER: Built-in entity recognition with refinement
   - **Status: Working but needs integration fixes**

4. **Data Models** (src/legacy_sentiment/data_models/)
   - Token dataclasses (Token, EntityToken, ProcessedToken)
   - Transcript structures (DialogueEntry, Section, TranscriptData)
   - **Status: Complete and stable**

5. **Custom Dictionaries** (data/language/)
   - custom_entities.json: 296 financial entities (companies, terms, products)
   - custom_mwe.json: Financial multi-word expressions
   - custom_regex_patterns.json: Financial patterns
   - custom_stops.json: 1,700+ custom stopwords
   - **Status: High quality, domain-specific, ready to use**

### ❌ Broken/Missing Components (Immediate Fixes Required)

1. **Missing Module: CustomEntityHandler**
   - Expected: src/legacy_sentiment/processing/custom_entity_handler.py
   - Available: superceded/custom_entity_handler.py (complete implementation)
   - Impact: EntityMWEHandler and SpaCyPipelineHandler fail on import
   - Priority: **HIGH**

2. **Missing Module: SemanticRoleHandler**
   - Expected: src/legacy_sentiment/nlp/semantic_role_handler.py
   - Available: superceded/semantic_role_handler.py (complete implementation)
   - Impact: Pipeline demos fail, semantic role analysis unavailable
   - Priority: **HIGH**

3. **Missing Module: unified_matcher_refactored**
   - Expected: src/legacy_sentiment/processing/unified_matcher_refactored.py
   - Available: superceded/unified_matcher_refactored.py (complete with fallback)
   - Impact: EnhancedSemanticRoleHandler uses fallback to superceded/
   - Priority: **MEDIUM** (fallback works but not ideal)

4. **Broken Imports** (Relative instead of absolute)
   - EntityMWEHandler.py lines 20-26: Missing package prefix
   - spacy_pipeline_handler.py lines 10-17: Missing package prefix
   - text_cleaner.py line 4: Missing package prefix
   - Impact: Modules cannot be imported, demos fail at startup
   - Priority: **HIGH**

5. **Streamlit Demos** (Both currently non-functional)
   - test_EntityMWEHandler.py: Fails on import
   - test_spacy_pipeline.py: Fails on import
   - Impact: No UI for testing/demo
   - Priority: **MEDIUM** (after fixing imports)

### ⚠️ Problematic Components (Reevaluate & Simplify)

Based on user feedback that "sophisticated linguistic analysis never delivered useful results," these components need critical evaluation:

1. **EnhancedSemanticRoleHandler** (src/legacy_sentiment/nlp/enhanced_semantic_role_handler.py)
   - Purpose: Extract semantic roles and complex structures
   - Issues: Overly complex, many custom rules, uncertain practical value
   - Status: **REEVALUATE** - May be over-engineered for use case

2. **SemanticRoleHandler** (superceded/semantic_role_handler.py)
   - Purpose: Basic predicate-argument structure extraction
   - Coverage: Core roles (AGENT, PATIENT, RECIPIENT), prep arguments, modifiers
   - Status: **REEVALUATE** - Simpler than enhanced version, may be sufficient

3. **Aspect Handler** (src/legacy_sentiment/utils/aspect_handler.py)
   - Purpose: Aspect-based sentiment analysis
   - Status: **REEVALUATE** - Determine if aspect extraction provides value

4. **Lexical Feature Analysis** (spacy_pipeline_handler.py lines 488-615)
   - Features: Uncertainty, negation, causal verbs, intensifiers, sarcasm
   - Complexity: POS validation, entity filtering, multi-category matching
   - Status: **REEVALUATE** - May be too granular for practical use

5. **unified_matcher_refactored** (superceded/)
   - Purpose: Complex token matching with span expansion
   - Complexity: 600+ lines, automaton-based matching, position tracking
   - Status: **REEVALUATE** - May be overkill if simpler handlers work

---

## Rebuild Procedure

### Phase 1: Restore Missing Modules (Days 1-2)

#### Task 1.1: Restore CustomEntityHandler
**Location:** Create `src/legacy_sentiment/processing/custom_entity_handler.py`

**Steps:**
1. Copy implementation from `superceded/custom_entity_handler.py`
2. Update imports:
   ```python
   # OLD (line 8)
   from custom_file_utils import load_custom_entities

   # NEW
   from legacy_sentiment.utils.custom_file_utils import load_custom_entities
   ```
3. Add to package exports in `src/legacy_sentiment/processing/__init__.py`
4. Verify: Run `python -c "from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler"`

**Dependencies:** None
**Estimated Time:** 30 minutes
**Risk:** Low (straightforward copy + import update)

---

#### Task 1.2: Restore SemanticRoleHandler
**Location:** Create `src/legacy_sentiment/nlp/semantic_role_handler.py`

**Steps:**
1. Copy implementation from `superceded/semantic_role_handler.py`
2. Update imports:
   ```python
   # OLD (line 8)
   from data_types import SemanticRole

   # NEW
   from legacy_sentiment.data_models.data_types import SemanticRole
   ```
3. Add to package exports in `src/legacy_sentiment/nlp/__init__.py`
4. Verify: Run `python -c "from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler"`

**Dependencies:** None
**Estimated Time:** 30 minutes
**Risk:** Low (straightforward copy + import update)

---

#### Task 1.3: Restore unified_matcher_refactored
**Location:** Create `src/legacy_sentiment/processing/unified_matcher_refactored.py`

**Steps:**
1. Copy implementation from `superceded/unified_matcher_refactored.py`
2. Update imports (lines 12-19):
   ```python
   # OLD
   from data_types import (
       Token,
       SPACY_MODEL_SM,
       SPACY_MODEL_MD,
       ...
   )

   # NEW
   from legacy_sentiment.data_models.data_types import (
       Token,
       SPACY_MODEL_SM,
       SPACY_MODEL_MD,
       ...
   )
   ```
3. Remove fallback import from `enhanced_semantic_role_handler.py` (lines 17-26):
   ```python
   # Remove the try/except fallback, just use:
   from legacy_sentiment.processing.unified_matcher_refactored import (
       get_excluded_positions,
       is_position_excluded,
   )
   ```
4. Add to package exports
5. Verify: Run `python -c "from legacy_sentiment.processing.unified_matcher_refactored import create_token"`

**Dependencies:** Task 1.1, Task 1.2
**Estimated Time:** 45 minutes
**Risk:** Medium (complex module, test thoroughly)

---

### Phase 2: Fix Broken Imports (Day 2)

#### Task 2.1: Fix EntityMWEHandler.py imports
**File:** `src/legacy_sentiment/processing/EntityMWEHandler.py`

**Changes (lines 20-26):**
```python
# OLD - Relative imports
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
from regex_pattern_handler import RegexPatternHandler
from spacy_handler import SpaCyHandler

# NEW - Absolute imports
from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
from legacy_sentiment.processing.mwe_handler import MWEHandler
from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
from legacy_sentiment.nlp.spacy_handler import SpaCyHandler
```

**Testing:**
```bash
python -c "from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler"
```

**Dependencies:** Task 1.1
**Estimated Time:** 15 minutes
**Risk:** Low

---

#### Task 2.2: Fix spacy_pipeline_handler.py imports
**File:** `src/legacy_sentiment/nlp/spacy_pipeline_handler.py`

**Changes (lines 10-17):**
```python
# OLD - Relative imports
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
from regex_pattern_handler import RegexPatternHandler
from semantic_role_handler import SemanticRoleHandler
from aspect_handler import AspectHandler

# NEW - Absolute imports
from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
from legacy_sentiment.processing.mwe_handler import MWEHandler
from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler
from legacy_sentiment.utils.aspect_handler import AspectHandler
```

**Testing:**
```bash
python -c "from legacy_sentiment.nlp.spacy_pipeline_handler import SpaCyPipelineHandler"
```

**Dependencies:** Task 1.1, Task 1.2
**Estimated Time:** 15 minutes
**Risk:** Low

---

#### Task 2.3: Fix text_cleaner.py imports
**File:** `src/legacy_sentiment/processing/text_cleaner.py`

**Changes (line 4):**
```python
# OLD - Relative import
from data_types import Token

# NEW - Absolute import
from legacy_sentiment.data_models.data_types import Token
```

**Testing:**
```bash
python -c "from legacy_sentiment.processing.text_cleaner import TextCleaner"
```

**Dependencies:** None
**Estimated Time:** 5 minutes
**Risk:** Low

---

### Phase 3: Verify Core Functionality (Day 3)

#### Task 3.1: Compilation Test
```bash
python -m compileall src/legacy_sentiment
```
**Expected:** No syntax errors
**If failures:** Fix syntax issues before proceeding

---

#### Task 3.2: Module Import Test
Create test script: `tests/test_imports.py`

```python
#!/usr/bin/env python3
"""Test all module imports work correctly."""

def test_data_models():
    from legacy_sentiment.data_models.data_types import Token, EntityToken
    from legacy_sentiment.data_models.transcript_structures import TranscriptData
    print("✓ Data models import successfully")

def test_ingestion():
    from legacy_sentiment.ingestion.transcript_handler import TranscriptHandler
    from legacy_sentiment.ingestion.json_transcript_parser import JSONTranscriptParser
    print("✓ Ingestion modules import successfully")

def test_processing():
    from legacy_sentiment.processing.preprocessing import TextPreprocessor
    from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler
    from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
    from legacy_sentiment.processing.mwe_handler import MWEHandler
    from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
    print("✓ Processing modules import successfully")

def test_nlp():
    from legacy_sentiment.nlp.spacy_handler import SpaCyHandler
    from legacy_sentiment.nlp.spacy_pipeline_handler import SpaCyPipelineHandler
    from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler
    from legacy_sentiment.nlp.enhanced_semantic_role_handler import EnhancedSemanticRoleHandler
    print("✓ NLP modules import successfully")

def test_utils():
    from legacy_sentiment.utils.custom_file_utils import load_custom_entities
    from legacy_sentiment.utils.stopword_handler import StopwordHandler
    from legacy_sentiment.utils.aspect_handler import AspectHandler
    print("✓ Utility modules import successfully")

if __name__ == "__main__":
    test_data_models()
    test_ingestion()
    test_processing()
    test_nlp()
    test_utils()
    print("\n✅ All imports successful!")
```

**Run:** `python tests/test_imports.py`
**Expected:** All checks pass

---

#### Task 3.3: Functional Test - Preprocessing Pipeline
Create test script: `tests/test_preprocessing.py`

```python
#!/usr/bin/env python3
"""Test preprocessing pipeline with sample data."""

from legacy_sentiment.processing.preprocessing import TextPreprocessor, PreprocessingConfig

def test_preprocessing():
    # Load configuration
    config = PreprocessingConfig.from_json('preprocessing_config.json')

    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)

    # Test text
    test_text = """
    Our Q3 revenue increased by approximately 15% year-over-year to $2.5 billion.
    EBITDA margins expanded to 28%, driven by operational efficiency improvements.
    """

    # Process
    result = preprocessor.preprocess_text(test_text)

    # Verify results
    assert result['cleaned_text'], "Cleaned text should not be empty"
    assert result['tokens'], "Tokens should be extracted"
    assert len(result['custom_patterns']) > 0, "Should detect financial patterns"

    print("✓ Preprocessing pipeline working correctly")
    print(f"  - Tokens extracted: {len(result['tokens'])}")
    print(f"  - Patterns found: {len(result['custom_patterns'])}")
    print(f"  - Stopwords removed: {result.get('stopwords_removed', 0)}")

    return True

if __name__ == "__main__":
    test_preprocessing()
    print("\n✅ Preprocessing test passed!")
```

**Run:** `python tests/test_preprocessing.py`
**Expected:** Test passes, shows metrics

---

#### Task 3.4: Functional Test - Entity Recognition
Create test script: `tests/test_entity_recognition.py`

```python
#!/usr/bin/env python3
"""Test entity recognition across all handlers."""

from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler

def test_entity_recognition():
    # Initialize handler with sample config
    handler = EntityMWEHandler(
        custom_entities_files=['data/language/custom_entities.json'],
        multi_word_entries_files=['data/language/custom_mwe.json'],
        regex_patterns_files=['data/language/custom_regex_patterns.json'],
        custom_stopwords_files=['data/language/custom_stops.json'],
        preprocessing_config={
            'tokenize': True,
            'extract_entities': True,
            'lemmatize': True
        }
    )

    # Test text with known entities
    test_text = """
    Apple Inc. reported strong earnings for Q3 2024, with revenue of $95.5 billion.
    The chief financial officer noted that approximately 40% of revenue came from iPhone sales.
    Year-over-year growth was 8.5%.
    """

    # Get active components
    components = handler.get_active_components()
    print("Active components:", components)

    # Process text
    results = handler.process_text(test_text)

    # Analyze results
    total_entities = 0
    entity_sources = {}

    for sentence, (entities, tokens, cleaned) in results:
        total_entities += len(entities)
        for entity, label, source, start, end in entities:
            entity_sources[source] = entity_sources.get(source, 0) + 1
            print(f"  Entity: '{entity}' | Type: {label} | Source: {source}")

    print(f"\n✓ Entity recognition working correctly")
    print(f"  - Total entities found: {total_entities}")
    print(f"  - By source: {entity_sources}")

    # Assertions
    assert total_entities > 0, "Should detect at least some entities"
    assert 'Custom Entity' in entity_sources or 'spaCy' in entity_sources, "Should use at least one handler"

    return True

if __name__ == "__main__":
    test_entity_recognition()
    print("\n✅ Entity recognition test passed!")
```

**Run:** `python tests/test_entity_recognition.py`
**Expected:** Detects entities from multiple sources

---

#### Task 3.5: Functional Test - Transcript Processing
Create test script: `tests/test_transcript_processing.py`

```python
#!/usr/bin/env python3
"""Test full transcript ingestion and processing."""

from legacy_sentiment.ingestion.transcript_handler import TranscriptHandler
from legacy_sentiment.processing.preprocessing import TextPreprocessor, PreprocessingConfig

def test_transcript_processing():
    # Initialize handler
    handler = TranscriptHandler()

    # Load sample transcript
    transcript_data = handler.load_transcript('data/transcripts/earnings_call_sample.json')

    assert transcript_data, "Should load transcript data"
    print(f"✓ Loaded transcript with {len(transcript_data.sections)} sections")

    # Initialize preprocessor
    config = PreprocessingConfig.from_json('preprocessing_config.json')
    preprocessor = TextPreprocessor(config)

    # Process each section
    total_dialogue_entries = 0
    for section in transcript_data.sections:
        print(f"\nSection: {section.title}")
        for dialogue in section.dialogue_entries:
            total_dialogue_entries += 1
            result = preprocessor.preprocess_text(dialogue.text)
            print(f"  - {dialogue.speaker}: {len(result['tokens'])} tokens")

    print(f"\n✓ Processed {total_dialogue_entries} dialogue entries")
    assert total_dialogue_entries > 0, "Should process at least one dialogue entry"

    return True

if __name__ == "__main__":
    test_transcript_processing()
    print("\n✅ Transcript processing test passed!")
```

**Run:** `python tests/test_transcript_processing.py`
**Expected:** Successfully processes sample transcript

---

### Phase 4: Test Streamlit Demos (Day 3)

#### Task 4.1: Test EntityMWEHandler Demo
```bash
streamlit run src/legacy_sentiment/streamlit/test_EntityMWEHandler.py
```

**Expected Behavior:**
- App launches without import errors
- Sidebar shows configuration options
- File upload works for transcripts
- Entity detection displays correctly
- Token analysis shows processed results

**If failures:** Debug import issues or handler initialization

---

#### Task 4.2: Test SpaCy Pipeline Demo
```bash
streamlit run src/legacy_sentiment/streamlit/test_spacy_pipeline.py
```

**Expected Behavior:**
- App launches without import errors
- Full pipeline analysis displays
- Semantic roles shown (if handler works)
- POS tags and noun chunks displayed
- Lexical features highlighted

**If failures:** Debug pipeline integration issues

---

### Phase 5: Evaluate Sophisticated Linguistic Analysis (Days 4-5)

**Critical Questions to Answer:**

1. **What was the original intent for linguistic analysis?**
   - Sentiment analysis at aspect level?
   - Semantic role labeling for information extraction?
   - Uncertainty/negation detection for risk analysis?

2. **Why did it underdeliver?**
   - Over-engineered for simple use case?
   - Inaccurate results (precision/recall issues)?
   - Too slow for production use?
   - Results not interpretable or actionable?

3. **What are the actual user requirements?**
   - What insights are needed from transcripts?
   - What decisions will this analysis support?
   - What level of accuracy is acceptable?

---

#### Task 5.1: Benchmark Semantic Role Handler (Basic)
Create test script: `tests/test_semantic_roles_basic.py`

```python
#!/usr/bin/env python3
"""Test basic semantic role handler."""

import spacy
from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler

def test_semantic_roles():
    nlp = spacy.load("en_core_web_sm")
    handler = SemanticRoleHandler(nlp)

    # Test sentences with clear semantic structure
    test_sentences = [
        "Apple increased revenue by 15% in Q3.",
        "The CFO announced a new cost reduction initiative.",
        "We expect margins to improve throughout fiscal 2024."
    ]

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        roles = handler.extract_roles_from_doc(nlp(sentence))

        for role in roles:
            print(f"  Predicate: {role.predicate} | Argument: {role.argument} | Role: {role.role}")

        if len(roles) == 0:
            print("  (No roles detected)")

    return True

if __name__ == "__main__":
    test_semantic_roles()
```

**Run:** `python tests/test_semantic_roles_basic.py`
**Evaluate:**
- Are roles accurate?
- Are they useful for information extraction?
- Do they provide actionable insights?

---

#### Task 5.2: Benchmark Enhanced Semantic Role Handler
Create test script: `tests/test_semantic_roles_enhanced.py`

```python
#!/usr/bin/env python3
"""Test enhanced semantic role handler with complexes."""

import spacy
from legacy_sentiment.nlp.enhanced_semantic_role_handler import EnhancedSemanticRoleHandler
from legacy_sentiment.processing.unified_matcher_refactored import unified_match, create_token
from legacy_sentiment.utils.custom_file_utils import load_custom_entities

def test_enhanced_semantic_roles():
    nlp = spacy.load("en_core_web_sm")
    handler = EnhancedSemanticRoleHandler(nlp)

    test_sentences = [
        "Revenue increased significantly in Q3 due to strong product demand.",
        "We may see potential headwinds from currency fluctuations.",
        "Margins did not improve as expected."
    ]

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        doc = nlp(sentence)

        # Get matches (simplified)
        matches = []  # Would need full unified_match setup

        # Extract roles and complexes
        results = handler.extract_roles_from_doc(doc, matches)

        for item in results:
            if hasattr(item, 'base'):  # SemanticComplex
                print(f"  Complex: {item.complex_type}")
                print(f"    Base: {item.base.argument} ({item.base.role})")
                if item.modifiers:
                    print(f"    Modifiers: {[m.argument for m in item.modifiers]}")
            else:  # EnhancedSemanticRole
                print(f"  Role: {item.argument} ({item.role})")

        if len(results) == 0:
            print("  (No semantic structures detected)")

    return True

if __name__ == "__main__":
    test_enhanced_semantic_roles()
```

**Run:** `python tests/test_semantic_roles_enhanced.py`
**Evaluate:**
- Does complexity add value?
- Are complexes accurately identified?
- Is the output interpretable?
- Does it justify the added code complexity?

---

#### Task 5.3: Benchmark Lexical Feature Analysis
Create test script: `tests/test_lexical_features.py`

```python
#!/usr/bin/env python3
"""Test lexical feature extraction."""

from legacy_sentiment.nlp.spacy_pipeline_handler import SpaCyPipelineHandler

def test_lexical_features():
    handler = SpaCyPipelineHandler(
        language_data_files='data/language/language_data.json'
    )

    # Test sentences with various features
    test_sentences = [
        "Revenue was approximately $2.5 billion.",  # Quantitative modifier
        "We may see potential headwinds.",  # Uncertainty
        "Performance did not meet expectations.",  # Negation
        "Growth was driven by strong demand.",  # Causal
        "Results were extremely positive."  # Intensifier
    ]

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        analysis = handler.analyze_text(sentence)

        features = analysis['lexical_features']
        for category, matches in features.items():
            if matches:
                print(f"  {category}:")
                for match in matches:
                    print(f"    - {match['term']} ({match['type']})")

    return True

if __name__ == "__main__":
    test_lexical_features()
```

**Run:** `python tests/test_lexical_features.py`
**Evaluate:**
- Are features accurately detected?
- Are they useful for sentiment/risk analysis?
- Is the granularity appropriate?

---

#### Task 5.4: Performance Benchmarking
Create test script: `tests/benchmark_performance.py`

```python
#!/usr/bin/env python3
"""Benchmark processing speed for different components."""

import time
from legacy_sentiment.processing.preprocessing import TextPreprocessor, PreprocessingConfig
from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler
from legacy_sentiment.nlp.spacy_pipeline_handler import SpaCyPipelineHandler

def benchmark_component(name, func, iterations=10):
    start = time.time()
    for _ in range(iterations):
        func()
    elapsed = time.time() - start
    avg = elapsed / iterations
    print(f"{name}: {avg:.4f}s per iteration ({iterations} iterations)")
    return avg

def test_sample_text():
    return """
    Our Q3 revenue increased by approximately 15% year-over-year to $2.5 billion.
    EBITDA margins expanded to 28%, driven by operational efficiency improvements.
    We expect continued growth throughout fiscal 2024, though we may face headwinds
    from currency fluctuations. The CFO noted that roughly 40% of revenue came from
    our core product line.
    """

def benchmark_preprocessing():
    config = PreprocessingConfig.from_json('preprocessing_config.json')
    preprocessor = TextPreprocessor(config)
    text = test_sample_text()
    preprocessor.preprocess_text(text)

def benchmark_entity_recognition():
    handler = EntityMWEHandler(
        custom_entities_files=['data/language/custom_entities.json'],
        multi_word_entries_files=['data/language/custom_mwe.json'],
        regex_patterns_files=['data/language/custom_regex_patterns.json'],
        preprocessing_config={'tokenize': True, 'extract_entities': True}
    )
    text = test_sample_text()
    handler.process_text(text)

def benchmark_full_pipeline():
    handler = SpaCyPipelineHandler(
        language_data_files='data/language/language_data.json',
        custom_entities_files='data/language/custom_entities.json',
        mwe_files='data/language/custom_mwe.json',
        regex_patterns_files='data/language/custom_regex_patterns.json'
    )
    text = test_sample_text()
    handler.analyze_text(text)

if __name__ == "__main__":
    print("Performance Benchmarking\n" + "="*50)
    benchmark_component("Preprocessing", benchmark_preprocessing)
    benchmark_component("Entity Recognition", benchmark_entity_recognition)
    benchmark_component("Full Pipeline", benchmark_full_pipeline)
    print("\n✅ Benchmarking complete!")
```

**Run:** `python tests/benchmark_performance.py`
**Evaluate:**
- Is performance acceptable for production use?
- Which components are bottlenecks?
- Can sophisticated components be optimized or removed?

---

### Phase 6: Recommendations & Simplification (Days 6-7)

Based on Phase 5 evaluation, create recommendations document.

#### Task 6.1: Create Analysis Report
Document: `LINGUISTIC_ANALYSIS_EVALUATION.md`

**Template:**
```markdown
# Linguistic Analysis Evaluation Report

## Executive Summary
[Brief overview of findings]

## Component Evaluations

### 1. Basic Semantic Role Handler
- **Accuracy:** [Results from testing]
- **Usefulness:** [Insights provided]
- **Performance:** [Speed metrics]
- **Recommendation:** KEEP / SIMPLIFY / REMOVE
- **Rationale:** [Justification]

### 2. Enhanced Semantic Role Handler
- **Accuracy:** [Results]
- **Added Value over Basic:** [Comparison]
- **Complexity Cost:** [Code complexity analysis]
- **Recommendation:** KEEP / SIMPLIFY / REMOVE
- **Rationale:** [Justification]

### 3. Lexical Feature Analysis
- **Accuracy:** [Results]
- **Usefulness:** [Practical applications]
- **Performance:** [Speed metrics]
- **Recommendation:** KEEP / SIMPLIFY / REMOVE
- **Rationale:** [Justification]

### 4. Aspect Handler
- **Accuracy:** [Results]
- **Usefulness:** [Value for sentiment analysis]
- **Recommendation:** KEEP / SIMPLIFY / REMOVE
- **Rationale:** [Justification]

### 5. Unified Matcher
- **Complexity:** [Code analysis]
- **Necessity:** [Can simpler approach work?]
- **Recommendation:** KEEP / SIMPLIFY / REMOVE
- **Rationale:** [Justification]

## Overall Recommendations

### Architecture Simplification
[Proposed simplified architecture]

### Components to Keep
- [Component 1]: [Why]
- [Component 2]: [Why]

### Components to Remove
- [Component 1]: [Why]
- [Component 2]: [Why]

### Components to Simplify
- [Component 1]: [How to simplify]
- [Component 2]: [How to simplify]

## Implementation Plan
[Steps for implementing recommendations]
```

---

#### Task 6.2: Proposed Simplified Architecture

If evaluation shows sophisticated components underdeliver, propose this simplified architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSCRIPT INGESTION                     │
│              (Keep as-is: JSON/TXT/PDF parsers)              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING (SIMPLIFIED)                 │
│  - Text cleaning                                            │
│  - Tokenization (NLTK)                                      │
│  - Stopword removal (with financial term protection)        │
│  - Lemmatization (conditional)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              ENTITY RECOGNITION (STREAMLINED)               │
│  - Custom entities (dictionary-based)                       │
│  - Multi-word expressions (Aho-Corasick)                    │
│  - Regex patterns (dates, currency, percentages)            │
│  - spaCy NER (basic, no complex refinement)                 │
│  - Overlap resolution (priority-based)                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           BASIC LINGUISTIC FEATURES (OPTIONAL)              │
│  - POS tagging (spaCy)                                      │
│  - Noun chunks (spaCy)                                      │
│  - Simple negation detection (dependency-based)             │
│  - [REMOVE: Complex semantic roles, aspect analysis]        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT & VISUALIZATION                         │
│  - Display by section/speaker (as-is)                       │
│  - Entity highlighting with source attribution             │
│  - Token-level analysis (lemma, POS, action)               │
│  - [REMOVE: Semantic role display, aspect display]          │
└─────────────────────────────────────────────────────────────┘
```

**Key Changes:**
1. **Remove:** EnhancedSemanticRoleHandler, SemanticComplex analysis
2. **Remove:** unified_matcher_refactored (use simpler handlers directly)
3. **Simplify:** Lexical feature analysis (keep only essential features)
4. **Optional:** Basic SemanticRoleHandler (only if proven valuable)
5. **Focus:** Entity recognition and display (what works well)

---

### Phase 7: Implementation of Updates (Days 8-10)

#### Option A: Keep Current Architecture (If evaluation is positive)

**Tasks:**
- Document best practices for using each component
- Create example notebooks for common use cases
- Optimize performance bottlenecks
- Add comprehensive testing

#### Option B: Implement Simplified Architecture (If evaluation is negative)

**Tasks:**

1. **Create simplified_processing package**
   - `src/legacy_sentiment/simplified/`
   - Copy working components (preprocessing, entity recognition)
   - Remove dependencies on semantic role handlers

2. **Create SimpleEntityPipeline class**
   ```python
   class SimpleEntityPipeline:
       def __init__(self, config):
           self.preprocessor = TextPreprocessor(config)
           self.entity_handler = CustomEntityHandler(...)
           self.mwe_handler = MWEHandler(...)
           self.regex_handler = RegexPatternHandler(...)
           self.spacy_nlp = spacy.load("en_core_web_sm")

       def process_text(self, text):
           # 1. Preprocess
           # 2. Extract entities from all sources
           # 3. Resolve overlaps
           # 4. Return results
           pass
   ```

3. **Create simplified Streamlit demo**
   - Focus on entity display and speaker/section analysis
   - Remove complex linguistic feature displays
   - Emphasize speed and clarity

4. **Deprecate complex components**
   - Move to `deprecated/` folder
   - Update documentation
   - Keep for reference but don't maintain

---

### Phase 8: Documentation & Testing (Days 11-12)

#### Task 8.1: Update Documentation

**Files to update:**
1. `README.md` - Reflect current architecture
2. `docs/ARCHITECTURE.md` - Document final design decisions
3. `docs/API_REFERENCE.md` - Document public APIs
4. `docs/USER_GUIDE.md` - How to use the system
5. `docs/DEVELOPER_GUIDE.md` - How to extend the system

#### Task 8.2: Create Comprehensive Test Suite

**Test categories:**
1. Unit tests for each module
2. Integration tests for pipelines
3. End-to-end tests with sample transcripts
4. Performance regression tests
5. Edge case tests (empty input, malformed data, etc.)

**Target coverage:** >80% for core modules

#### Task 8.3: Create Example Notebooks

**Notebooks:**
1. `01_preprocessing_tutorial.ipynb` - Text preprocessing walkthrough
2. `02_entity_recognition_tutorial.ipynb` - Entity extraction examples
3. `03_transcript_analysis_tutorial.ipynb` - Full transcript analysis
4. `04_custom_dictionaries_tutorial.ipynb` - How to customize vocabularies
5. `05_integration_tutorial.ipynb` - Integrating into production systems

---

### Phase 9: Deployment Preparation (Days 13-14)

#### Task 9.1: Package Setup

**Create proper Python package:**
```bash
# Setup tools
pip install build twine

# Create pyproject.toml
# Create setup.py
# Add MANIFEST.in

# Build package
python -m build

# Test installation
pip install dist/legacy_sentiment-*.whl
```

#### Task 9.2: Dependencies Management

**Create requirements files:**
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `requirements-streamlit.txt` - Streamlit demo dependencies

**Pin versions:**
```
spacy==3.7.2
nltk==3.8.1
pyahocorasick==2.0.0
streamlit==1.29.0
```

#### Task 9.3: CI/CD Setup

**GitHub Actions workflows:**
1. `.github/workflows/tests.yml` - Run tests on push
2. `.github/workflows/lint.yml` - Code quality checks
3. `.github/workflows/build.yml` - Package building

#### Task 9.4: Docker Setup (Optional)

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY data/ ./data/
COPY preprocessing_config.json .

CMD ["streamlit", "run", "src/legacy_sentiment/streamlit/test_EntityMWEHandler.py"]
```

---

## Risk Assessment

### High Risk Items

1. **Semantic Role Handler Integration**
   - Risk: May not work as expected after restoration
   - Mitigation: Thorough testing with sample transcripts
   - Contingency: Use fallback to simpler extraction

2. **Performance at Scale**
   - Risk: Processing large transcripts may be slow
   - Mitigation: Benchmark with realistic data volumes
   - Contingency: Implement caching, parallel processing

3. **Dictionary Maintenance**
   - Risk: Custom entities may become outdated
   - Mitigation: Create process for regular updates
   - Contingency: Add automated entity suggestion based on frequency

### Medium Risk Items

1. **spaCy Model Updates**
   - Risk: Newer spaCy versions may change behavior
   - Mitigation: Pin spaCy version, test before upgrades
   - Contingency: Lock to known-good version

2. **Streamlit Demo Complexity**
   - Risk: Demos may be too complex for users
   - Mitigation: Gather user feedback, iterate
   - Contingency: Create simpler demo version

### Low Risk Items

1. **Import Path Changes**
   - Risk: Breaking existing code
   - Mitigation: This is a rebuild, no legacy users
   - Contingency: N/A

---

## Success Criteria

### Phase 1-4 (Core Restoration)
- ✅ All modules import without errors
- ✅ Compilation test passes
- ✅ Both Streamlit demos launch successfully
- ✅ Entity recognition works across all handlers
- ✅ Preprocessing pipeline produces expected output

### Phase 5 (Evaluation)
- ✅ Documented accuracy metrics for linguistic analysis
- ✅ Performance benchmarks for all components
- ✅ Clear recommendation document with rationale

### Phase 6-7 (Optimization)
- ✅ Implemented recommended architecture changes
- ✅ Removed or simplified underperforming components
- ✅ Updated demos reflect final architecture

### Phase 8 (Documentation)
- ✅ Comprehensive documentation for all public APIs
- ✅ Test coverage >80% for core modules
- ✅ Tutorial notebooks for common use cases

### Phase 9 (Deployment)
- ✅ Installable Python package
- ✅ CI/CD pipeline functioning
- ✅ Docker container builds and runs

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Restore Missing Modules | 1 day | 3 modules restored, imports fixed |
| Phase 2: Fix Broken Imports | 0.5 days | All imports absolute, no errors |
| Phase 3: Verify Core Functionality | 1 day | Test suite passes, demos work |
| Phase 4: Test Streamlit Demos | 0.5 days | Both demos functional |
| Phase 5: Evaluate Linguistics | 2 days | Benchmarks, accuracy metrics |
| Phase 6: Recommendations | 1 day | Evaluation report, architecture proposal |
| Phase 7: Implementation | 3 days | Updated architecture implemented |
| Phase 8: Documentation & Testing | 2 days | Full documentation, tests >80% |
| Phase 9: Deployment Prep | 2 days | Packaged, CI/CD, Docker |
| **Total** | **13-14 days** | **Production-ready system** |

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Clarify requirements** for linguistic analysis (what's actually needed?)
3. **Begin Phase 1** - restore missing modules
4. **Schedule checkpoints** after Phases 3, 5, and 7 for review
5. **Iterate based on evaluation** - be prepared to pivot in Phase 6

---

## Questions to Answer Before Starting

1. **What is the primary use case for this system?**
   - Extracting financial metrics from transcripts?
   - Sentiment analysis of speaker statements?
   - Risk identification (uncertainty, negation)?
   - Competitive intelligence?

2. **Who are the end users?**
   - Financial analysts?
   - Researchers?
   - Automated trading systems?
   - Content reviewers?

3. **What defines success for the linguistic analysis?**
   - Accuracy threshold?
   - Speed requirements?
   - Interpretability of results?
   - Actionable insights generated?

4. **What was the original dissatisfaction with linguistic analysis?**
   - False positives/negatives?
   - Too complex to interpret?
   - Too slow?
   - Not relevant to business needs?

5. **Are there budget/time constraints?**
   - Should we prioritize speed over comprehensiveness?
   - Is this a proof-of-concept or production system?

---

## Appendix: Key File Locations

### Missing Modules (Source)
- `superceded/custom_entity_handler.py` → restore to `src/legacy_sentiment/processing/`
- `superceded/semantic_role_handler.py` → restore to `src/legacy_sentiment/nlp/`
- `superceded/unified_matcher_refactored.py` → restore to `src/legacy_sentiment/processing/`

### Files with Broken Imports
- `src/legacy_sentiment/processing/EntityMWEHandler.py` (lines 20-26)
- `src/legacy_sentiment/nlp/spacy_pipeline_handler.py` (lines 10-17)
- `src/legacy_sentiment/processing/text_cleaner.py` (line 4)

### Configuration Files
- `preprocessing_config.json` - Preprocessing settings
- `data/language/*.json` - Custom dictionaries

### Sample Data
- `data/transcripts/earnings_call_sample.json`
- `data/transcripts/earnings_call_sample.txt`

### Demo Applications
- `src/legacy_sentiment/streamlit/test_EntityMWEHandler.py`
- `src/legacy_sentiment/streamlit/test_spacy_pipeline.py`

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Author:** Legacy Sentiment Rebuild Team
