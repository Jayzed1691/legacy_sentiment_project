"""
Sentiment Analyzer for Earnings Call Transcripts

A focused, production-ready system for analyzing sentiment and generating
speaker feedback from earnings call transcripts.

Key Features:
- Topic extraction from financial discussions
- LLM-based sentiment analysis (via Ollama)
- Speaker-level feedback generation
- Clean, maintainable architecture

Modules:
- core: Transcript parsing and context analysis
- extractors: Entity, multi-word, and pattern extraction
- analyzers: Sentiment, confidence, and tone analysis
- generators: Feedback and report generation
- integrations: Ollama LLM integration
- utils: File loading, text processing utilities
- data: Data models and configuration
"""

__version__ = "2.0.0"
__author__ = "Legacy Sentiment Team"

from sentiment_analyzer.core.transcript import TranscriptParser
from sentiment_analyzer.extractors.entity import EntityExtractor
from sentiment_analyzer.extractors.multiword import MultiwordExtractor
from sentiment_analyzer.extractors.pattern import PatternExtractor

__all__ = [
    "TranscriptParser",
    "EntityExtractor",
    "MultiwordExtractor",
    "PatternExtractor",
]
