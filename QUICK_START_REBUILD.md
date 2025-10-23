# Quick Start: Legacy Sentiment Project Rebuild

## TL;DR - Immediate Actions

This repository has **3 missing modules** and **3 files with broken imports** that need immediate fixing before the system can function.

---

## Critical Issues Summary

### ❌ Missing Modules (In `superceded/` folder, need to be restored)

1. **CustomEntityHandler** → Required by EntityMWEHandler and SpaCyPipelineHandler
2. **SemanticRoleHandler** → Required by SpaCyPipelineHandler and demos
3. **unified_matcher_refactored** → Required by EnhancedSemanticRoleHandler (currently using fallback)

### ❌ Broken Imports (Using relative instead of absolute imports)

1. **EntityMWEHandler.py** (lines 20-26)
2. **spacy_pipeline_handler.py** (lines 10-17)
3. **text_cleaner.py** (line 4)

---

## What Works Right Now ✅

- **Transcript Ingestion:** JSON and TXT parsers
- **Preprocessing:** Text cleaning, tokenization, lemmatization, stopword removal
- **Entity Recognition:** Custom entities, MWE, regex patterns (handlers exist)
- **Data Models:** All dataclasses complete
- **Custom Dictionaries:** 296 financial entities, MWEs, regex patterns, 1700+ stopwords

---

## What Doesn't Work ❌

- **Streamlit Demos:** Both fail on import errors
- **Full Pipeline:** Can't initialize EntityMWEHandler or SpaCyPipelineHandler
- **Entity Orchestration:** Missing CustomEntityHandler breaks coordination
- **Semantic Analysis:** Missing SemanticRoleHandler prevents semantic extraction

---

## Fast Fix (1-2 Hours)

### Step 1: Restore CustomEntityHandler (15 min)

```bash
# Copy file to correct location
cp superceded/custom_entity_handler.py src/legacy_sentiment/processing/custom_entity_handler.py

# Edit line 8 in the new file
# Change: from custom_file_utils import load_custom_entities
# To:     from legacy_sentiment.utils.custom_file_utils import load_custom_entities
```

### Step 2: Restore SemanticRoleHandler (15 min)

```bash
# Copy file to correct location
cp superceded/semantic_role_handler.py src/legacy_sentiment/nlp/semantic_role_handler.py

# Edit line 8 in the new file
# Change: from data_types import SemanticRole
# To:     from legacy_sentiment.data_models.data_types import SemanticRole
```

### Step 3: Restore unified_matcher_refactored (20 min)

```bash
# Copy file to correct location
cp superceded/unified_matcher_refactored.py src/legacy_sentiment/processing/unified_matcher_refactored.py

# Edit lines 12-19 in the new file
# Change: from data_types import (
# To:     from legacy_sentiment.data_models.data_types import (
```

### Step 4: Fix EntityMWEHandler.py imports (5 min)

Edit `src/legacy_sentiment/processing/EntityMWEHandler.py` lines 20-23:

```python
# OLD (lines 20-23)
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
from regex_pattern_handler import RegexPatternHandler
from spacy_handler import SpaCyHandler

# NEW (lines 20-23)
from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
from legacy_sentiment.processing.mwe_handler import MWEHandler
from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
from legacy_sentiment.nlp.spacy_handler import SpaCyHandler
```

### Step 5: Fix spacy_pipeline_handler.py imports (5 min)

Edit `src/legacy_sentiment/nlp/spacy_pipeline_handler.py` lines 10-16:

```python
# OLD (lines 10-16)
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
from regex_pattern_handler import RegexPatternHandler
from semantic_role_handler import SemanticRoleHandler
from aspect_handler import AspectHandler

# NEW (lines 10-16)
from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
from legacy_sentiment.processing.mwe_handler import MWEHandler
from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler
from legacy_sentiment.utils.aspect_handler import AspectHandler
```

### Step 6: Fix text_cleaner.py import (2 min)

Edit `src/legacy_sentiment/processing/text_cleaner.py` line 4:

```python
# OLD (line 4)
from data_types import Token

# NEW (line 4)
from legacy_sentiment.data_models.data_types import Token
```

### Step 7: Fix enhanced_semantic_role_handler.py import (5 min)

Edit `src/legacy_sentiment/nlp/enhanced_semantic_role_handler.py` lines 17-26:

Remove the try/except fallback and use direct import:

```python
# OLD (lines 17-26)
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

# NEW (lines 17-22)
from legacy_sentiment.processing.unified_matcher_refactored import (
        get_excluded_positions,
        is_position_excluded,
)
```

### Step 8: Verify (10 min)

```bash
# Test compilation
python -m compileall src/legacy_sentiment

# Test imports
python -c "from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler; print('✓ CustomEntityHandler')"
python -c "from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler; print('✓ SemanticRoleHandler')"
python -c "from legacy_sentiment.processing.unified_matcher_refactored import create_token; print('✓ unified_matcher_refactored')"
python -c "from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler; print('✓ EntityMWEHandler')"
python -c "from legacy_sentiment.nlp.spacy_pipeline_handler import SpaCyPipelineHandler; print('✓ SpaCyPipelineHandler')"

# Test demo (should launch without errors)
streamlit run src/legacy_sentiment/streamlit/test_EntityMWEHandler.py
```

---

## After Fast Fix - Next Steps

Once the system is functional, proceed with evaluation:

### 1. Test Core Functionality (30 min)

```bash
# Test preprocessing
python -c "
from legacy_sentiment.processing.preprocessing import TextPreprocessor, PreprocessingConfig
config = PreprocessingConfig.from_json('preprocessing_config.json')
preprocessor = TextPreprocessor(config)
result = preprocessor.preprocess_text('Apple Inc. revenue increased by 15% to \$2.5 billion.')
print(f'Tokens: {len(result[\"tokens\"])}')
print(f'Patterns: {len(result[\"custom_patterns\"])}')
"

# Test entity recognition
python -c "
from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler
handler = EntityMWEHandler(
    custom_entities_files=['data/language/custom_entities.json'],
    multi_word_entries_files=['data/language/custom_mwe.json'],
    regex_patterns_files=['data/language/custom_regex_patterns.json']
)
results = handler.process_text('Apple Inc. Q3 revenue was \$95.5 billion, up 8.5% YoY.')
for sentence, (entities, tokens, cleaned) in results:
    print(f'Entities: {len(entities)}')
    for entity, label, source, start, end in entities:
        print(f'  {entity} ({label}) from {source}')
"
```

### 2. Evaluate Sophisticated Linguistics (1-2 hours)

**Key Question:** Why did sophisticated linguistic analysis underdeliver?

Test semantic role extraction:
```bash
python -c "
import spacy
from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler
nlp = spacy.load('en_core_web_sm')
handler = SemanticRoleHandler(nlp)
roles = handler.extract_roles_from_doc(nlp('Revenue increased by 15% in Q3.'))
for role in roles:
    print(f'{role.predicate} | {role.argument} | {role.role}')
"
```

**Questions to answer:**
- Are the extracted roles accurate?
- Are they useful for your analysis goals?
- Do they provide insights you couldn't get from simpler methods?
- Is the performance acceptable?

### 3. Decide on Architecture (Based on evaluation)

**Option A: Keep sophisticated analysis** if it proves valuable
- Document usage patterns
- Optimize performance
- Create tutorials

**Option B: Simplify architecture** if it underdelivers
- Remove EnhancedSemanticRoleHandler
- Remove SemanticComplex structures
- Remove unified_matcher_refactored
- Focus on entity recognition + basic display

---

## File Structure Reference

```
legacy_sentiment_project/
├── src/legacy_sentiment/
│   ├── processing/
│   │   ├── EntityMWEHandler.py          ← FIX IMPORTS (lines 20-26)
│   │   ├── custom_entity_handler.py     ← RESTORE from superceded/
│   │   ├── unified_matcher_refactored.py ← RESTORE from superceded/
│   │   ├── text_cleaner.py              ← FIX IMPORT (line 4)
│   │   ├── preprocessing.py             ✅ WORKING
│   │   ├── mwe_handler.py               ✅ WORKING
│   │   └── regex_pattern_handler.py     ✅ WORKING
│   ├── nlp/
│   │   ├── spacy_pipeline_handler.py    ← FIX IMPORTS (lines 10-17)
│   │   ├── semantic_role_handler.py     ← RESTORE from superceded/
│   │   ├── enhanced_semantic_role_handler.py ← FIX IMPORT (lines 17-26)
│   │   ├── spacy_handler.py             ✅ WORKING
│   │   └── named_entity_recognition.py  ✅ WORKING
│   ├── ingestion/                       ✅ ALL WORKING
│   ├── data_models/                     ✅ ALL WORKING
│   └── utils/                           ✅ ALL WORKING
├── superceded/                          ← SOURCE for missing modules
│   ├── custom_entity_handler.py
│   ├── semantic_role_handler.py
│   └── unified_matcher_refactored.py
├── data/
│   ├── language/                        ✅ ALL READY
│   └── transcripts/                     ✅ SAMPLES READY
├── docs/
│   └── reconstruction_status.md         ← Original recovery notes
├── REBUILD_PLAN.md                      ← Comprehensive 14-day plan
└── QUICK_START_REBUILD.md              ← This file
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'custom_entity_handler'"
**Solution:** You didn't restore the module. Go back to Step 1.

### Issue: "ImportError: cannot import name 'load_custom_entities'"
**Solution:** You didn't update the imports after restoring. Edit the file to use absolute imports.

### Issue: "ModuleNotFoundError: No module named 'legacy_sentiment'"
**Solution:** You're not running from the project root, or the package isn't installed. Run from `/home/user/legacy_sentiment_project/` or install the package with `pip install -e .`

### Issue: Streamlit demo shows blank page
**Solution:** Check the browser console for errors. Likely an import issue wasn't fully resolved.

### Issue: "LookupError: [E050] Can't find model 'en_core_web_sm'"
**Solution:** Download the spaCy model: `python -m spacy download en_core_web_sm`

---

## Success Indicators

After completing the fast fix, you should see:

✅ `python -m compileall src/legacy_sentiment` - No errors
✅ All import tests pass
✅ Streamlit demos launch without errors
✅ Entity recognition detects companies, financial terms, dates, percentages
✅ Preprocessing produces tokens and patterns
✅ Transcript parsing works for JSON and TXT files

---

## What to Do After Fast Fix

1. **Read REBUILD_PLAN.md** for comprehensive 14-day rebuild procedure
2. **Run Phase 3 tests** to verify core functionality
3. **Run Phase 5 evaluation** to assess sophisticated linguistic components
4. **Decide on architecture** based on your actual requirements
5. **Implement Phase 6-7** changes based on evaluation
6. **Complete Phase 8-9** for production readiness

---

## Questions?

Refer to:
- **REBUILD_PLAN.md** - Complete rebuild procedure with all phases
- **docs/reconstruction_status.md** - Original recovery notes
- **readme.md** - Project overview and features

---

**Estimated Time:**
- Fast Fix: 1-2 hours
- Testing & Evaluation: 2-4 hours
- Architecture Decision: 1 hour
- Full Rebuild (if following complete plan): 13-14 days
