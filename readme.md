# Legacy Sentiment Project

Legacy Sentiment is an NLP toolkit focused on financial transcripts. It layers custom dictionaries, spaCy pipelines, and semantic-role extraction on top of ingest utilities so analysts can explore earnings calls with Streamlit dashboards or standalone scripts.

## Features
- **Configurable preprocessing** – The NLTK-based `TextPreprocessor` normalizes text, preserves financial vocabulary, and extracts regex-driven patterns while reading settings from `preprocessing_config.json`.
- **Custom spaCy pipeline** – `EntityMWEHandler` and the spaCy helpers load domain dictionaries, multi-word expressions, and semantic-role metadata to annotate transcripts with consistent spans.
- **Transcript ingestion** – Parsers convert JSON, TXT, and PDF call notes into structured `TranscriptData` objects that power the demo applications and downstream analytics.
- **Streamlit demos** – Two dashboards (`test_EntityMWEHandler.py` and `test_spacy_pipeline.py`) showcase entity detection, preprocessing, and transcript browsing.

## Repository Layout
```
├── data/
│   ├── language/                # Sample dictionaries for entities, stopwords, regex rules, and semantic roles
│   └── transcripts/             # Example transcripts ready for upload to the demos
├── preprocessing_config.json    # Default preprocessing settings consumed by the apps and helpers
├── src/legacy_sentiment/
│   ├── data_models/             # Token, semantic-role, and transcript data structures
│   ├── ingestion/               # JSON/TXT/PDF transcript parsers plus convenience wrappers
│   ├── nlp/                     # spaCy pipeline setup, semantic-role handler, and NER utilities
│   ├── processing/              # Text preprocessor, entity + MWE handlers, regex helpers, and cleaners
│   ├── streamlit/               # UI components for uploading transcripts and running analyses
│   └── utils/                   # File-loading utilities, stopword helpers, and aspect configuration logic
└── superceded/                  # Archived legacy modules kept for reference only
```

## Configuration and Sample Data
The repository includes ready-to-use assets so the demos work immediately:

- **`preprocessing_config.json`** – Controls the `TextPreprocessor` pipeline. It specifies language, boolean flags (cleaning, stopword removal, lemmatization), and the sample dictionary files in `data/language/`. Streamlit writes back to this file when configuration changes are saved.
- **Language dictionaries (`data/language/`)** – JSON resources that power entity recognition, multi-word detection, regex extraction, and stopword overrides. They can be replaced with organization-specific vocabularies while keeping the same schema.
- **Sample transcripts (`data/transcripts/`)** – Includes `earnings_call_sample.json` for structured ingestion and `earnings_call_sample.txt` for plain-text parsing.

### Transcript JSON Format
JSON transcripts should follow the structure expected by `JSONTranscriptParser`:
```json
{
  "transcript": [
    {
      "section": "Prepared Remarks",
      "speakers": [
        {"name": "Speaker", "role": "Role", "dialogue": "Utterance text"}
      ],
      "subsections": [
        {
          "section": "Optional Subsection",
          "speakers": [...],
          "subsections": []
        }
      ]
    }
  ]
}
```
Each section contains speaker entries (name, optional role, and dialogue text) plus optional nested subsections for deeper hierarchies.

## Running the Demos
1. Create a virtual environment and install dependencies (spaCy, nltk, ahocorasick, streamlit).
2. Download the required spaCy model, e.g. `python -m spacy download en_core_web_sm`.
3. Launch the Streamlit interfaces from the project root:
   - `streamlit run src/legacy_sentiment/streamlit/test_EntityMWEHandler.py`
   - `streamlit run src/legacy_sentiment/streamlit/test_spacy_pipeline.py`
4. Use the sidebar controls to load the bundled configuration and transcripts or upload your own files.

## Testing
A quick smoke check can be performed with:
```
python -m compileall src/legacy_sentiment
```
Add automated tests under `tests/` to cover transcript ingestion, preprocessing, and pipeline integration as new functionality stabilizes.
