# Reconstruction Status

This document tracks modules that are still missing from the recovered codebase and can be recreated based on surviving references and archived implementations.

## High-Priority Gaps

### 1. `CustomEntityHandler`
* **Where it is referenced:** The active entity and pipeline orchestrators import `CustomEntityHandler` without a package prefix, so they currently depend on a module that does not exist under `src/legacy_sentiment`.
  * `EntityMWEHandler` expects it while wiring custom dictionaries for sentence processing.【F:src/legacy_sentiment/processing/EntityMWEHandler.py†L12-L26】【F:src/legacy_sentiment/processing/EntityMWEHandler.py†L66-L76】
  * `SpaCyPipelineHandler` also imports it to enrich spaCy outputs with custom matches.【F:src/legacy_sentiment/nlp/spacy_pipeline_handler.py†L5-L18】
* **What survives:** A full implementation lives in the `superceded` archive and only needs path updates plus integration with the modern loader utilities.【F:superceded/custom_entity_handler.py†L5-L55】
* **Reconstruction scope:** Move the legacy class into `src/legacy_sentiment/processing/custom_entity_handler.py`, swap its file-loading calls to `legacy_sentiment.utils.custom_file_utils`, and update imports to use the package-qualified path.

### 2. `SemanticRoleHandler`
* **Where it is referenced:** The Streamlit demo imports `legacy_sentiment.nlp.semantic_role_handler`, but no such module ships with the reorganized package.【F:src/legacy_sentiment/streamlit/test_EntityMWEHandler.py†L24-L33】
* **What survives:** A working handler that extracts predicate-argument structures is archived in `superceded/semantic_role_handler.py` and still aligns with the current data models once the imports are corrected.【F:superceded/semantic_role_handler.py†L12-L148】
* **Reconstruction scope:** Port the class into `src/legacy_sentiment/nlp/semantic_role_handler.py`, refactor it to import `SemanticRole` from the new dataclass module, and expose it via the `nlp` package `__all__`.

### 3. `unified_matcher_refactored`
* **Where it is referenced:** The enhanced semantic-role handler attempts to import `legacy_sentiment.processing.unified_matcher_refactored` and falls back to the superceded module when that fails.【F:src/legacy_sentiment/nlp/enhanced_semantic_role_handler.py†L11-L26】
* **What survives:** The full matcher engine with token creation utilities remains in `superceded/unified_matcher_refactored.py` and matches the expectations of the handler (helpers like `get_excluded_positions`).【F:superceded/unified_matcher_refactored.py†L1-L120】
* **Reconstruction scope:** Promote the matcher into `src/legacy_sentiment/processing/unified_matcher_refactored.py`, update its imports to pull token dataclasses from `legacy_sentiment.data_models`, and remove the legacy fallback.

## Additional Notes
* After recreating these modules, fix the in-package imports (e.g., use `from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler`) so the code can be executed without relying on the archived directory.
* Re-run `python -m compileall src/legacy_sentiment` once the modules are restored to confirm there are no syntax regressions.
