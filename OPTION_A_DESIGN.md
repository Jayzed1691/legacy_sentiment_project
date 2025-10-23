# Option A: Simple Sentiment Approach with LLM Enhancement

## Executive Summary

This design implements a **topic-level sentiment analysis system** for earnings call speaker feedback, leveraging:
- ✅ Existing entity extraction (proven working)
- ✅ Local LLMs via Ollama (modern, private, cost-free)
- ✅ Improved naming conventions (professional, maintainable)
- ✅ Focused architecture (only what's needed for the use case)

**Goal:** Show each speaker how their statements might be received in the financial marketplace.

---

## Architecture Overview

### Simplified Pipeline

```
┌─────────────────────┐
│ EARNINGS CALL       │
│ TRANSCRIPT          │
│ (JSON/TXT)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ TRANSCRIPT PARSER                               │
│ - Load transcript structure                     │
│ - Split by speaker and section                  │
│ - Preserve metadata (time, role, etc.)          │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ TOPIC EXTRACTOR                                 │
│ - Custom entity recognition ✅ (working)        │
│ - Multi-word expressions ✅ (working)           │
│ - Regex patterns ✅ (working)                   │
│ - Group by semantic topics (e.g., "revenue     │
│   growth", "margin expansion")                  │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ CONTEXT ANALYZER                                │
│ - Extract 1-2 sentences around each topic      │
│ - Preserve speaker identity                     │
│ - Include temporal context (Q3 2024, YoY)      │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ SENTIMENT ANALYZER (Ollama LLM)                 │
│ - Analyze sentiment: positive/negative/neutral  │
│ - Detect confidence level: confident/cautious   │
│ - Identify tone: optimistic/defensive/balanced  │
│ - Extract key points mentioned                  │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ FEEDBACK GENERATOR (Ollama LLM)                 │
│ - Aggregate by speaker                          │
│ - Compare to best practices                     │
│ - Generate actionable recommendations           │
│ - Highlight strengths and opportunities         │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ REPORT OUTPUT                                   │
│ - Speaker-level summaries                       │
│ - Topic-level analysis                          │
│ - Comparative insights (vs. past calls)         │
│ - Export to PDF/HTML/JSON                       │
└─────────────────────────────────────────────────┘
```

---

## Why Ollama LLMs Are Perfect for This

### Advantages

1. **Local & Private**
   - Earnings call data is sensitive
   - No data sent to external APIs
   - HIPAA/SOC2 compliant if needed

2. **Cost-Free**
   - No per-token pricing
   - Unlimited analysis
   - Process entire call history

3. **Financial Understanding**
   - Modern LLMs (Llama 3, Mistral) understand financial language
   - Can detect nuance ("strong" vs "solid" vs "robust")
   - Context-aware (knows "headwinds" is negative even without explicit negative words)

4. **Natural Language Output**
   - Generate human-readable feedback
   - Explain reasoning
   - Provide examples and recommendations

5. **Fast Iteration**
   - Test different models (Llama 3.1, Mistral, Phi-3)
   - Switch models based on performance
   - No vendor lock-in

### Recommended Models

**For Sentiment Analysis:**
- **Llama 3.1 8B** - Best balance of speed and accuracy
- **Mistral 7B** - Slightly faster, good for financial text
- **Phi-3 Medium** - Fastest, acceptable accuracy

**For Feedback Generation:**
- **Llama 3.1 70B** (if hardware permits) - Most sophisticated
- **Mixtral 8x7B** - Excellent reasoning, good recommendations
- **Llama 3.1 8B** - Sufficient for clear feedback

### Sample Prompts

**Sentiment Analysis:**
```
You are a financial analyst evaluating an earnings call statement.

Statement: "{speaker} said: {context}"
Topic: {topic}

Analyze:
1. Sentiment (positive/negative/neutral)
2. Confidence level (confident/cautious/uncertain)
3. Tone (optimistic/defensive/balanced/concerned)
4. Key point being made

Respond in JSON:
{
  "sentiment": "positive|negative|neutral",
  "confidence": "confident|cautious|uncertain",
  "tone": "optimistic|defensive|balanced|concerned",
  "key_point": "brief summary",
  "reasoning": "why you assessed this way"
}
```

**Feedback Generation:**
```
You are a communications coach for investor relations professionals.

Speaker: {speaker_name} ({role})
Topics discussed: {topics_list}

Analysis:
{topic_sentiment_data}

Generate actionable feedback:
1. What did they do well?
2. What could be improved?
3. How might the market receive their statements?
4. Specific recommendations for next earnings call

Be direct, specific, and constructive. Focus on communication effectiveness.
```

---

## Improved Naming Conventions

### Current Issues
- Mixed case: `EntityMWEHandler.py` (PascalCase) vs `custom_entity_handler.py` (snake_case)
- Unclear purpose: `processing/` contains both preprocessing and entity handling
- Redundant names: `custom_entity_handler.py` has `CustomEntityHandler` class
- Legacy naming: `spacy_handler.py` when spaCy is optional

### Proposed Convention

**Python Standard: snake_case for files, PascalCase for classes**

#### Module Organization
```
src/sentiment_analyzer/
├── core/
│   ├── __init__.py
│   ├── transcript.py          # TranscriptParser, TranscriptData
│   ├── topic.py               # TopicExtractor, Topic
│   └── context.py             # ContextAnalyzer, Context
│
├── extractors/
│   ├── __init__.py
│   ├── entity.py              # EntityExtractor (was CustomEntityHandler)
│   ├── multiword.py           # MultiwordExtractor (was MWEHandler)
│   ├── pattern.py             # PatternExtractor (was RegexPatternHandler)
│   └── topic.py               # TopicExtractor (new - groups entities into topics)
│
├── analyzers/
│   ├── __init__.py
│   ├── sentiment.py           # SentimentAnalyzer (new - Ollama-based)
│   ├── confidence.py          # ConfidenceDetector (new)
│   └── tone.py                # ToneAnalyzer (new)
│
├── generators/
│   ├── __init__.py
│   ├── feedback.py            # FeedbackGenerator (new - Ollama-based)
│   └── report.py              # ReportGenerator (new - PDF/HTML output)
│
├── integrations/
│   ├── __init__.py
│   ├── ollama.py              # OllamaClient (new)
│   └── prompts.py             # PromptTemplates (new)
│
├── utils/
│   ├── __init__.py
│   ├── file_loader.py         # FileLoader (was custom_file_utils)
│   ├── stopwords.py           # StopwordFilter (was stopword_handler)
│   └── text_cleaner.py        # TextCleaner (already exists)
│
└── data/
    ├── __init__.py
    ├── models.py              # Data models (Topic, Sentiment, Feedback)
    └── config.py              # Configuration classes
```

#### File Naming Rules

1. **Files:** `snake_case.py`
2. **Classes:** `PascalCase`
3. **Functions:** `snake_case()`
4. **Constants:** `UPPER_SNAKE_CASE`
5. **Private:** `_leading_underscore`

**Examples:**
```python
# extractors/entity.py
class EntityExtractor:
    def extract_entities(self, text: str) -> List[Entity]:
        ...

# analyzers/sentiment.py
class SentimentAnalyzer:
    def analyze_sentiment(self, context: Context) -> Sentiment:
        ...
```

#### Import Patterns

**Old (problematic):**
```python
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
```

**New (clear):**
```python
from sentiment_analyzer.extractors.entity import EntityExtractor
from sentiment_analyzer.extractors.multiword import MultiwordExtractor
```

**Even better (aliased):**
```python
from sentiment_analyzer.extractors import (
    EntityExtractor,
    MultiwordExtractor,
    PatternExtractor,
)
```

### Benefits

1. **Consistency** - All files follow Python conventions
2. **Clarity** - Purpose clear from module path
3. **Maintainability** - Easy to find and update
4. **Professional** - Matches industry standards
5. **Scalability** - Clear where new features go

---

## Implementation Plan

### Phase 1: Core Infrastructure (Day 1)

**1.1 Set up Ollama Integration**
```python
# integrations/ollama.py
import requests
from typing import Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        return response.json()["response"]

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate structured JSON response."""
        response = self.generate(prompt, temperature=0.3)
        # Parse JSON from response
        return json.loads(response)
```

**1.2 Create Prompt Templates**
```python
# integrations/prompts.py
class PromptTemplates:
    SENTIMENT_ANALYSIS = """
    You are a financial analyst evaluating an earnings call statement.

    Speaker: {speaker} ({role})
    Topic: {topic}
    Statement: "{context}"

    Analyze this statement and respond in JSON format:
    {{
      "sentiment": "positive|negative|neutral",
      "confidence": "confident|cautious|uncertain",
      "tone": "optimistic|defensive|balanced|concerned",
      "key_point": "one sentence summary",
      "reasoning": "why you assessed this way"
    }}
    """

    FEEDBACK_GENERATION = """
    You are a communications coach for investor relations professionals.

    Speaker: {speaker_name} ({role})
    Total statements analyzed: {statement_count}

    Topic Analysis:
    {topic_data}

    Generate actionable feedback in this format:
    1. Strengths (2-3 specific points)
    2. Areas for Improvement (2-3 specific points)
    3. Market Reception (how statements likely to be received)
    4. Recommendations for Next Call (3-4 actionable items)

    Be direct, specific, and constructive.
    """
```

**1.3 Design Data Models**
```python
# data/models.py
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ConfidenceLevel(Enum):
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    UNCERTAIN = "uncertain"

class ToneType(Enum):
    OPTIMISTIC = "optimistic"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    CONCERNED = "concerned"

@dataclass
class Topic:
    """Extracted financial topic"""
    name: str
    category: str
    entities: List[str]
    start: int
    end: int

@dataclass
class Context:
    """Context around a topic mention"""
    topic: Topic
    speaker: str
    role: Optional[str]
    text: str
    sentence: str
    timestamp: Optional[str]

@dataclass
class Sentiment:
    """Sentiment analysis result"""
    sentiment: SentimentType
    confidence: ConfidenceLevel
    tone: ToneType
    key_point: str
    reasoning: str
    score: float  # -1.0 to 1.0

@dataclass
class SpeakerAnalysis:
    """Aggregated analysis for a speaker"""
    speaker: str
    role: Optional[str]
    topics: List[Topic]
    sentiments: List[Sentiment]
    overall_sentiment: SentimentType
    overall_confidence: ConfidenceLevel
    topic_breakdown: Dict[str, List[Sentiment]]

@dataclass
class Feedback:
    """Generated feedback for a speaker"""
    speaker: str
    role: Optional[str]
    strengths: List[str]
    improvements: List[str]
    market_reception: str
    recommendations: List[str]
    overall_assessment: str
```

### Phase 2: Core Analyzers (Day 2)

**2.1 Sentiment Analyzer**
```python
# analyzers/sentiment.py
from sentiment_analyzer.integrations.ollama import OllamaClient
from sentiment_analyzer.integrations.prompts import PromptTemplates
from sentiment_analyzer.data.models import Context, Sentiment, SentimentType, ConfidenceLevel, ToneType
import json

class SentimentAnalyzer:
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    def analyze(self, context: Context) -> Sentiment:
        """Analyze sentiment of a topic mention in context."""
        prompt = PromptTemplates.SENTIMENT_ANALYSIS.format(
            speaker=context.speaker,
            role=context.role or "Unknown",
            topic=context.topic.name,
            context=context.sentence
        )

        try:
            response = self.client.generate_json(prompt)

            return Sentiment(
                sentiment=SentimentType(response["sentiment"]),
                confidence=ConfidenceLevel(response["confidence"]),
                tone=ToneType(response["tone"]),
                key_point=response["key_point"],
                reasoning=response["reasoning"],
                score=self._calculate_score(response)
            )
        except Exception as e:
            # Fallback to neutral if LLM fails
            return Sentiment(
                sentiment=SentimentType.NEUTRAL,
                confidence=ConfidenceLevel.UNCERTAIN,
                tone=ToneType.BALANCED,
                key_point="Analysis unavailable",
                reasoning=f"Error: {str(e)}",
                score=0.0
            )

    def _calculate_score(self, response: dict) -> float:
        """Convert categorical sentiment to numeric score."""
        sentiment_scores = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }

        confidence_weights = {
            "confident": 1.0,
            "cautious": 0.6,
            "uncertain": 0.3
        }

        base_score = sentiment_scores.get(response["sentiment"], 0.0)
        weight = confidence_weights.get(response["confidence"], 0.5)

        return base_score * weight
```

**2.2 Topic Extractor (groups entities into topics)**
```python
# extractors/topic.py
from sentiment_analyzer.extractors.entity import EntityExtractor
from sentiment_analyzer.extractors.multiword import MultiwordExtractor
from sentiment_analyzer.extractors.pattern import PatternExtractor
from sentiment_analyzer.data.models import Topic
from typing import List, Set

class TopicExtractor:
    def __init__(self,
                 entity_extractor: EntityExtractor,
                 multiword_extractor: MultiwordExtractor,
                 pattern_extractor: PatternExtractor):
        self.entity_extractor = entity_extractor
        self.multiword_extractor = multiword_extractor
        self.pattern_extractor = pattern_extractor

        # Define topic groupings
        self.topic_mappings = {
            "revenue": {"revenue", "sales", "income", "top line", "bookings"},
            "margins": {"margin", "EBITDA", "operating margin", "gross margin"},
            "growth": {"growth", "expansion", "increase", "year-over-year", "YoY"},
            "guidance": {"guidance", "outlook", "forecast", "expect", "anticipate"},
            "costs": {"costs", "expenses", "headcount", "efficiency"},
            "market": {"market", "competition", "competitive", "market share"},
            "product": {"product", "offering", "solution", "platform"},
        }

    def extract_topics(self, text: str) -> List[Topic]:
        """Extract topics from text using entity extraction."""
        # Get all entities
        entities = self.entity_extractor.extract_entities(text)
        mwes = self.multiword_extractor.extract_multiword_expressions(text)
        patterns = self.pattern_extractor.extract_patterns(text)

        # Group into topics
        topics = []
        seen_positions = set()

        for entity_text, category, source, start, end in entities + mwes + patterns:
            if (start, end) in seen_positions:
                continue

            topic_name = self._map_to_topic(entity_text.lower(), category)
            if topic_name:
                topics.append(Topic(
                    name=topic_name,
                    category=category,
                    entities=[entity_text],
                    start=start,
                    end=end
                ))
                seen_positions.add((start, end))

        return self._merge_nearby_topics(topics)

    def _map_to_topic(self, entity: str, category: str) -> Optional[str]:
        """Map an entity to a high-level topic."""
        for topic, keywords in self.topic_mappings.items():
            if any(keyword in entity for keyword in keywords):
                return topic
        return category if category in ["FINANCIAL_VOCABULARY", "TIME_PERIOD"] else None

    def _merge_nearby_topics(self, topics: List[Topic], proximity: int = 50) -> List[Topic]:
        """Merge topics that are close together in the text."""
        if not topics:
            return []

        merged = []
        current = topics[0]

        for next_topic in topics[1:]:
            if next_topic.start - current.end <= proximity and next_topic.name == current.name:
                # Merge
                current.entities.extend(next_topic.entities)
                current.end = next_topic.end
            else:
                merged.append(current)
                current = next_topic

        merged.append(current)
        return merged
```

### Phase 3: Feedback Generation (Day 3)

**3.1 Feedback Generator**
```python
# generators/feedback.py
from sentiment_analyzer.integrations.ollama import OllamaClient
from sentiment_analyzer.integrations.prompts import PromptTemplates
from sentiment_analyzer.data.models import SpeakerAnalysis, Feedback
from typing import Dict, List

class FeedbackGenerator:
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    def generate_feedback(self, analysis: SpeakerAnalysis) -> Feedback:
        """Generate actionable feedback for a speaker."""

        # Prepare topic data
        topic_data = self._format_topic_data(analysis)

        prompt = PromptTemplates.FEEDBACK_GENERATION.format(
            speaker_name=analysis.speaker,
            role=analysis.role or "Unknown",
            statement_count=len(analysis.sentiments),
            topic_data=topic_data
        )

        response = self.client.generate(prompt, temperature=0.7)

        # Parse response into structured feedback
        return self._parse_feedback(response, analysis)

    def _format_topic_data(self, analysis: SpeakerAnalysis) -> str:
        """Format topic analysis for prompt."""
        lines = []
        for topic, sentiments in analysis.topic_breakdown.items():
            positive = sum(1 for s in sentiments if s.sentiment.value == "positive")
            negative = sum(1 for s in sentiments if s.sentiment.value == "negative")
            neutral = sum(1 for s in sentiments if s.sentiment.value == "neutral")

            avg_score = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0

            lines.append(f"""
{topic.upper()}:
- Mentions: {len(sentiments)}
- Sentiment: {positive} positive, {neutral} neutral, {negative} negative
- Avg Score: {avg_score:.2f}
- Key Points: {', '.join(s.key_point for s in sentiments[:3])}
            """)

        return "\n".join(lines)

    def _parse_feedback(self, response: str, analysis: SpeakerAnalysis) -> Feedback:
        """Parse LLM response into structured feedback."""
        # Simple parsing (can be improved with structured output)
        sections = {
            "strengths": [],
            "improvements": [],
            "market_reception": "",
            "recommendations": []
        }

        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            if "strengths" in line.lower():
                current_section = "strengths"
            elif "improvement" in line.lower():
                current_section = "improvements"
            elif "market reception" in line.lower():
                current_section = "market_reception"
            elif "recommendation" in line.lower():
                current_section = "recommendations"
            elif current_section and line.startswith(('-', '•', '*', str)):
                content = line.lstrip('-•*0123456789. ')
                if current_section == "market_reception":
                    sections[current_section] += content + " "
                else:
                    sections[current_section].append(content)

        return Feedback(
            speaker=analysis.speaker,
            role=analysis.role,
            strengths=sections["strengths"],
            improvements=sections["improvements"],
            market_reception=sections["market_reception"].strip(),
            recommendations=sections["recommendations"],
            overall_assessment=self._generate_overall(analysis)
        )

    def _generate_overall(self, analysis: SpeakerAnalysis) -> str:
        """Generate overall assessment."""
        avg_score = sum(s.score for s in analysis.sentiments) / len(analysis.sentiments)

        if avg_score > 0.5:
            return "Strong, confident communication with positive market messaging"
        elif avg_score > 0:
            return "Generally positive communication with room for improvement"
        elif avg_score > -0.5:
            return "Balanced communication, consider strengthening key messages"
        else:
            return "Defensive communication, recommend refocusing on strengths"
```

### Phase 4: End-to-End Pipeline (Day 4)

**4.1 Main Pipeline**
```python
# core/pipeline.py
from sentiment_analyzer.core.transcript import TranscriptParser
from sentiment_analyzer.extractors.topic import TopicExtractor
from sentiment_analyzer.core.context import ContextAnalyzer
from sentiment_analyzer.analyzers.sentiment import SentimentAnalyzer
from sentiment_analyzer.generators.feedback import FeedbackGenerator
from sentiment_analyzer.data.models import SpeakerAnalysis, Feedback
from typing import Dict, List

class SentimentPipeline:
    """End-to-end pipeline for earnings call analysis."""

    def __init__(self,
                 transcript_parser: TranscriptParser,
                 topic_extractor: TopicExtractor,
                 context_analyzer: ContextAnalyzer,
                 sentiment_analyzer: SentimentAnalyzer,
                 feedback_generator: FeedbackGenerator):
        self.transcript_parser = transcript_parser
        self.topic_extractor = topic_extractor
        self.context_analyzer = context_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.feedback_generator = feedback_generator

    def process_transcript(self, transcript_path: str) -> Dict[str, Feedback]:
        """Process entire transcript and generate feedback for each speaker."""

        # 1. Parse transcript
        transcript_data = self.transcript_parser.parse(transcript_path)

        # 2. Process each speaker
        speaker_analyses = {}

        for speaker in transcript_data.get_unique_speakers():
            # Get all text from this speaker
            speaker_texts = transcript_data.get_speaker_text(speaker)

            # Extract topics from all their statements
            all_topics = []
            all_sentiments = []

            for text_segment in speaker_texts:
                # Extract topics
                topics = self.topic_extractor.extract_topics(text_segment.text)

                # Get context for each topic
                for topic in topics:
                    context = self.context_analyzer.get_context(
                        text_segment.text,
                        topic,
                        speaker=speaker,
                        role=text_segment.role
                    )

                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.analyze(context)

                    all_topics.append(topic)
                    all_sentiments.append(sentiment)

            # Aggregate by topic
            topic_breakdown = {}
            for topic, sentiment in zip(all_topics, all_sentiments):
                if topic.name not in topic_breakdown:
                    topic_breakdown[topic.name] = []
                topic_breakdown[topic.name].append(sentiment)

            # Create speaker analysis
            analysis = SpeakerAnalysis(
                speaker=speaker,
                role=speaker_texts[0].role if speaker_texts else None,
                topics=all_topics,
                sentiments=all_sentiments,
                overall_sentiment=self._get_overall_sentiment(all_sentiments),
                overall_confidence=self._get_overall_confidence(all_sentiments),
                topic_breakdown=topic_breakdown
            )

            speaker_analyses[speaker] = analysis

        # 3. Generate feedback for each speaker
        feedbacks = {}
        for speaker, analysis in speaker_analyses.items():
            feedbacks[speaker] = self.feedback_generator.generate_feedback(analysis)

        return feedbacks

    def _get_overall_sentiment(self, sentiments: List[Sentiment]) -> SentimentType:
        """Calculate overall sentiment."""
        avg_score = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0
        if avg_score > 0.2:
            return SentimentType.POSITIVE
        elif avg_score < -0.2:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL

    def _get_overall_confidence(self, sentiments: List[Sentiment]) -> ConfidenceLevel:
        """Calculate overall confidence level."""
        confident_count = sum(1 for s in sentiments if s.confidence == ConfidenceLevel.CONFIDENT)
        ratio = confident_count / len(sentiments) if sentiments else 0

        if ratio > 0.6:
            return ConfidenceLevel.CONFIDENT
        elif ratio > 0.3:
            return ConfidenceLevel.CAUTIOUS
        else:
            return ConfidenceLevel.UNCERTAIN
```

### Phase 5: Refactoring (Days 5-6)

**5.1 Migration Strategy**

```bash
# Create new structure
mkdir -p src/sentiment_analyzer/{core,extractors,analyzers,generators,integrations,utils,data}

# Copy and rename files (with imports updated)
# Old → New
src/legacy_sentiment/processing/custom_entity_handler.py → src/sentiment_analyzer/extractors/entity.py
src/legacy_sentiment/processing/mwe_handler.py → src/sentiment_analyzer/extractors/multiword.py
src/legacy_sentiment/processing/regex_pattern_handler.py → src/sentiment_analyzer/extractors/pattern.py

# Update all imports in each file
# Example: from custom_entity_handler import CustomEntityHandler
#       → from sentiment_analyzer.extractors.entity import EntityExtractor
```

**5.2 Automated Refactoring Script**
```python
# scripts/refactor_imports.py
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    "from custom_entity_handler import CustomEntityHandler":
        "from sentiment_analyzer.extractors.entity import EntityExtractor",
    "from mwe_handler import MWEHandler":
        "from sentiment_analyzer.extractors.multiword import MultiwordExtractor",
    # ... more mappings
}

def refactor_file(file_path: Path):
    """Update imports in a file."""
    content = file_path.read_text()

    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)

    file_path.write_text(content)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_sentiment_analyzer.py
def test_sentiment_analysis_positive():
    analyzer = SentimentAnalyzer(mock_ollama_client)
    context = Context(
        topic=Topic("revenue", "FINANCIAL", ["revenue"], 0, 7),
        speaker="CFO",
        role="Chief Financial Officer",
        text="Revenue grew 20% year-over-year",
        sentence="Revenue grew 20% year-over-year"
    )

    result = analyzer.analyze(context)

    assert result.sentiment == SentimentType.POSITIVE
    assert result.confidence in [ConfidenceLevel.CONFIDENT, ConfidenceLevel.CAUTIOUS]
    assert result.score > 0
```

### Integration Tests
```python
# tests/test_pipeline.py
def test_end_to_end_pipeline():
    pipeline = SentimentPipeline(...)

    feedbacks = pipeline.process_transcript("data/transcripts/earnings_call_sample.json")

    assert len(feedbacks) > 0
    assert "CFO" in feedbacks or "Chief Financial Officer" in feedbacks

    cfo_feedback = feedbacks.get("CFO") or feedbacks.get("Chief Financial Officer")
    assert cfo_feedback.strengths
    assert cfo_feedback.recommendations
```

---

## Migration Checklist

- [ ] Set up Ollama locally (install + pull llama3.1)
- [ ] Create new `sentiment_analyzer` package structure
- [ ] Implement Ollama integration
- [ ] Port existing extractors to new naming convention
- [ ] Implement sentiment analyzer
- [ ] Implement feedback generator
- [ ] Create end-to-end pipeline
- [ ] Write tests
- [ ] Test with real transcript
- [ ] Generate sample report
- [ ] Document API
- [ ] Update README with new architecture

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Infrastructure | 4 hours | Ollama client, prompts, data models |
| Phase 2: Analyzers | 6 hours | Sentiment analysis working |
| Phase 3: Generation | 4 hours | Feedback generation working |
| Phase 4: Pipeline | 4 hours | End-to-end pipeline |
| Phase 5: Refactoring | 8 hours | Clean codebase, new structure |
| Phase 6: Testing | 4 hours | Tests passing, validation |
| **Total** | **30 hours** | **Production-ready system** |

---

## Next Immediate Steps

1. **Install Ollama** (5 min)
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1
   ```

2. **Test Ollama** (5 min)
   ```bash
   ollama run llama3.1 "Analyze this financial statement sentiment: Revenue increased 20%"
   ```

3. **Create new package structure** (10 min)
   ```bash
   mkdir -p src/sentiment_analyzer/{core,extractors,analyzers,generators,integrations,utils,data}
   touch src/sentiment_analyzer/__init__.py
   # ... create __init__.py in each subdirectory
   ```

4. **Implement Ollama client** (30 min)
   - `integrations/ollama.py`
   - `integrations/prompts.py`

5. **Test sentiment analysis** (30 min)
   - Create simple test with hardcoded context
   - Verify Ollama responds correctly

**Ready to proceed?** I can start implementing immediately or we can discuss any adjustments to the design first.
