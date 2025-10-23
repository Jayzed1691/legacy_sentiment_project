"""
Data Models for Sentiment Analysis

This module defines the core data structures used throughout the sentiment
analysis pipeline, including transcripts, topics, sentiment, and feedback.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from collections import defaultdict


# =============================================================================
# Transcript Data Models
# =============================================================================

@dataclass
class DialogueEntry:
    """
    Represents a single speaker's dialogue in a transcript.

    Attributes:
        speaker: Name of the speaker
        role: Role/title of the speaker (e.g., "CEO", "CFO")
        text: The actual dialogue text
        sentiment_analysis: Optional sentiment analysis results for this dialogue
    """
    speaker: str
    role: str
    text: str
    sentiment_analysis: Optional['SentimentAnalysis'] = None


@dataclass
class Section:
    """
    Represents a section in a transcript (e.g., "Opening Remarks", "Q&A").

    Supports hierarchical structure with subsections.

    Attributes:
        name: Name of the section
        dialogues: List of dialogue entries in this section
        subsections: List of nested subsections
    """
    name: str
    dialogues: List[DialogueEntry] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)

    def all_dialogues(self) -> List[DialogueEntry]:
        """Get all dialogues including those in subsections."""
        dialogues = list(self.dialogues)
        for subsection in self.subsections:
            dialogues.extend(subsection.all_dialogues())
        return dialogues


@dataclass
class TranscriptData:
    """
    Complete transcript data structure.

    Attributes:
        sections: List of top-level sections
        speakers: Dictionary mapping speaker names to their dialogue texts
        metadata: Optional metadata (company name, date, etc.)
    """
    sections: List[Section] = field(default_factory=list)
    speakers: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    metadata: Dict[str, str] = field(default_factory=dict)

    def all_dialogues(self) -> List[DialogueEntry]:
        """Get all dialogues from all sections."""
        dialogues = []
        for section in self.sections:
            dialogues.extend(section.all_dialogues())
        return dialogues

    def get_speaker_dialogues(self, speaker: str) -> List[DialogueEntry]:
        """Get all dialogues for a specific speaker."""
        return [d for d in self.all_dialogues() if d.speaker == speaker]


# =============================================================================
# Sentiment Analysis Data Models
# =============================================================================

class SentimentType(str, Enum):
    """Sentiment polarity classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ConfidenceLevel(str, Enum):
    """Confidence level for sentiment classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ToneType(str, Enum):
    """Tone/manner of communication."""
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    OPTIMISTIC = "optimistic"
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"


@dataclass
class Topic:
    """
    Represents a topic extracted from text.

    Example topics: "revenue growth", "operating margins", "guidance"

    Attributes:
        text: The topic phrase/text
        category: Category of the topic (e.g., "financial", "operational")
        mentions: List of text snippets where this topic is mentioned
        start_pos: Starting position in source text
        end_pos: Ending position in source text
    """
    text: str
    category: str
    mentions: List[str] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class Context:
    """
    Contextual information surrounding a topic.

    Attributes:
        topic: The topic being discussed
        surrounding_text: Text around the topic for context
        entities: Related entities (companies, products, metrics)
        patterns: Financial patterns (currency, percentages, dates)
        speaker: Who is discussing this topic
        section: Which section of the transcript
    """
    topic: Topic
    surrounding_text: str
    entities: List[tuple] = field(default_factory=list)  # (text, category, source, start, end)
    patterns: List[tuple] = field(default_factory=list)  # (text, label, source, start, end)
    speaker: str = ""
    section: str = ""


@dataclass
class Sentiment:
    """
    Sentiment analysis result for a topic.

    Attributes:
        sentiment_type: Positive, negative, neutral, or mixed
        confidence: Confidence level of the classification
        tone: Tone of the communication
        reasoning: Natural language explanation of the sentiment
        score: Numerical sentiment score (-1.0 to 1.0)
        key_phrases: Important phrases that influenced the sentiment
    """
    sentiment_type: SentimentType
    confidence: ConfidenceLevel
    tone: ToneType
    reasoning: str
    score: float = 0.0
    key_phrases: List[str] = field(default_factory=list)


@dataclass
class SentimentAnalysis:
    """
    Complete sentiment analysis for a piece of text.

    Attributes:
        text: Original text analyzed
        topics: List of topics found in the text
        contexts: Context for each topic
        sentiments: Sentiment analysis for each topic
        overall_sentiment: Aggregate sentiment across all topics
    """
    text: str
    topics: List[Topic] = field(default_factory=list)
    contexts: List[Context] = field(default_factory=list)
    sentiments: Dict[str, Sentiment] = field(default_factory=dict)  # topic.text -> Sentiment
    overall_sentiment: Optional[Sentiment] = None


@dataclass
class Feedback:
    """
    Generated feedback for a speaker based on sentiment analysis.

    Attributes:
        speaker: Name of the speaker
        role: Role of the speaker
        summary: High-level summary of how statements were received
        strengths: Positive aspects highlighted
        concerns: Areas that may be received negatively
        suggestions: Actionable recommendations for improvement
        topic_breakdown: Per-topic sentiment breakdown
    """
    speaker: str
    role: str
    summary: str
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    topic_breakdown: Dict[str, Sentiment] = field(default_factory=dict)


@dataclass
class SpeakerAnalysis:
    """
    Complete analysis for a single speaker.

    Attributes:
        speaker: Speaker name
        role: Speaker role
        dialogues: All dialogue entries from this speaker
        sentiment_analyses: Sentiment analysis for each dialogue
        feedback: Generated feedback
        key_topics: Most important topics discussed
    """
    speaker: str
    role: str
    dialogues: List[DialogueEntry] = field(default_factory=list)
    sentiment_analyses: List[SentimentAnalysis] = field(default_factory=list)
    feedback: Optional[Feedback] = None
    key_topics: List[Topic] = field(default_factory=list)
