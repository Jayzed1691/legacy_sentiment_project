"""
Prompt Templates for LLM Interactions

Structured prompts for sentiment analysis, topic extraction, and feedback generation.
"""

from typing import List, Dict


class PromptTemplates:
    """Collection of prompt templates for financial sentiment analysis."""

    # =========================================================================
    # System Prompts
    # =========================================================================

    FINANCIAL_ANALYST = """You are an expert financial analyst specializing in earnings call analysis.
Your role is to analyze corporate communications and provide insights on how statements might
be received by investors and analysts in the financial marketplace.

Key principles:
- Focus on topics and themes, not individual words
- Consider context and tone carefully
- Identify both explicit and implicit sentiment
- Be objective and evidence-based
- Provide actionable insights"""

    INVESTOR_RELATIONS_ADVISOR = """You are an investor relations and communications advisor.
Your role is to help corporate executives improve their messaging for earnings calls
and investor communications.

Key principles:
- Provide constructive, actionable feedback
- Balance positive recognition with areas for improvement
- Consider audience perspective (investors, analysts, media)
- Focus on clarity, transparency, and confidence
- Suggest specific improvements"""

    # =========================================================================
    # Topic Extraction
    # =========================================================================

    @staticmethod
    def extract_topics(text: str, max_topics: int = 10) -> str:
        """Generate prompt for topic extraction."""
        return f"""Analyze the following earnings call dialogue and extract the main topics discussed.

DIALOGUE:
{text}

Extract up to {max_topics} key topics. For each topic, provide:
1. Topic name (brief, 2-5 words)
2. Category (e.g., "financial", "operational", "strategic", "guidance")
3. Key phrases that indicate this topic

Return your response as a JSON array with this structure:
[
    {{
        "topic": "revenue growth",
        "category": "financial",
        "key_phrases": ["revenue reached", "top-line growth", "sales increased"]
    }}
]"""

    # =========================================================================
    # Sentiment Analysis
    # =========================================================================

    @staticmethod
    def analyze_sentiment(topic: str, context: str, entities: List[str] = None,
                         patterns: List[str] = None) -> str:
        """Generate prompt for topic-level sentiment analysis."""
        entities_text = ""
        if entities:
            entities_text = f"\n\nRELATED ENTITIES: {', '.join(entities)}"

        patterns_text = ""
        if patterns:
            patterns_text = f"\n\nFINANCIAL PATTERNS: {', '.join(patterns)}"

        return f"""Analyze the sentiment expressed about the topic "{topic}" in the following context.

CONTEXT:
{context}{entities_text}{patterns_text}

Provide a detailed sentiment analysis including:
1. Sentiment type (positive, negative, neutral, or mixed)
2. Confidence level (high, medium, low)
3. Tone (confident, cautious, optimistic, defensive, neutral)
4. Detailed reasoning for your assessment
5. Sentiment score (-1.0 to 1.0, where -1.0 is very negative, 0 is neutral, 1.0 is very positive)
6. Key phrases that influenced your assessment

Return your response as JSON with this structure:
{{
    "sentiment_type": "positive",
    "confidence": "high",
    "tone": "confident",
    "reasoning": "The speaker uses strong positive language...",
    "score": 0.8,
    "key_phrases": ["exceeded expectations", "strong performance"]
}}"""

    # =========================================================================
    # Feedback Generation
    # =========================================================================

    @staticmethod
    def generate_speaker_feedback(speaker: str, role: str,
                                  topics: List[Dict], sentiments: Dict[str, Dict]) -> str:
        """Generate prompt for speaker-level feedback."""
        # Build topics summary
        topics_summary = []
        for topic_dict in topics[:5]:  # Top 5 topics
            topic_name = topic_dict.get('topic', 'Unknown')
            sentiment = sentiments.get(topic_name, {})
            sent_type = sentiment.get('sentiment_type', 'unknown')
            score = sentiment.get('score', 0)
            topics_summary.append(f"- {topic_name}: {sent_type} (score: {score:.2f})")

        topics_text = "\n".join(topics_summary)

        return f"""Generate constructive feedback for {speaker} ({role}) based on their earnings call performance.

KEY TOPICS AND SENTIMENT:
{topics_text}

Provide comprehensive feedback including:
1. Summary: High-level assessment of how their statements likely were received
2. Strengths: 2-3 specific positive aspects (what worked well)
3. Concerns: 2-3 areas that may have been received negatively
4. Suggestions: 2-3 actionable recommendations for improvement

Focus on:
- How statements might be perceived by investors and analysts
- Balance between transparency and confidence
- Clarity of communication
- Handling of sensitive topics (if any)

Return your response as JSON with this structure:
{{
    "summary": "Overall assessment...",
    "strengths": [
        "Clear articulation of revenue growth strategy",
        "Transparent discussion of challenges"
    ],
    "concerns": [
        "Defensive tone when discussing margins",
        "Vague guidance for next quarter"
    ],
    "suggestions": [
        "Provide more specific metrics for margin improvement",
        "Frame challenges as opportunities with clear action plans"
    ]
}}"""

    # =========================================================================
    # Overall Analysis
    # =========================================================================

    @staticmethod
    def analyze_overall_sentiment(all_topics: List[Dict], all_sentiments: Dict) -> str:
        """Generate prompt for overall transcript sentiment."""
        topics_summary = []
        for topic_dict in all_topics:
            topic_name = topic_dict.get('topic', 'Unknown')
            sentiment = all_sentiments.get(topic_name, {})
            sent_type = sentiment.get('sentiment_type', 'unknown')
            score = sentiment.get('score', 0)
            topics_summary.append(f"- {topic_name}: {sent_type} (score: {score:.2f})")

        topics_text = "\n".join(topics_summary)

        return f"""Analyze the overall sentiment of this earnings call based on topic-level analysis.

TOPIC-LEVEL SENTIMENTS:
{topics_text}

Provide an overall sentiment assessment including:
1. Dominant sentiment type across all topics
2. Overall confidence level
3. Prevailing tone
4. Comprehensive reasoning
5. Overall sentiment score (-1.0 to 1.0)
6. Most impactful topics (positive and negative)

Return your response as JSON with this structure:
{{
    "sentiment_type": "mixed",
    "confidence": "medium",
    "tone": "cautiously optimistic",
    "reasoning": "While revenue topics were positive...",
    "score": 0.3,
    "key_phrases": ["strong revenue", "margin pressure", "guidance maintained"]
}}"""

    # =========================================================================
    # Context Building
    # =========================================================================

    @staticmethod
    def build_context_summary(text: str, entities: List[tuple],
                             patterns: List[tuple]) -> str:
        """Build a context summary for sentiment analysis."""
        context_parts = [text]

        if entities:
            entity_strs = [f"{e[0]} ({e[1]})" for e in entities[:10]]
            context_parts.append(f"\nMentioned: {', '.join(entity_strs)}")

        if patterns:
            pattern_strs = [f"{p[0]} ({p[1]})" for p in patterns[:10]]
            context_parts.append(f"\nFinancial data: {', '.join(pattern_strs)}")

        return "\n".join(context_parts)

    # =========================================================================
    # Few-Shot Examples
    # =========================================================================

    SENTIMENT_EXAMPLE_POSITIVE = {
        "context": "Our revenue reached $125 million, exceeding expectations by 15%. We're seeing strong momentum across all product lines.",
        "analysis": {
            "sentiment_type": "positive",
            "confidence": "high",
            "tone": "confident",
            "reasoning": "Clear positive indicators: 'exceeded expectations', specific strong percentage (15%), 'strong momentum'. No hedging or qualifiers.",
            "score": 0.9,
            "key_phrases": ["exceeded expectations", "strong momentum", "$125 million"]
        }
    }

    SENTIMENT_EXAMPLE_NEGATIVE = {
        "context": "Margins declined by 200 basis points due to increased costs. We're working on efficiency improvements.",
        "analysis": {
            "sentiment_type": "negative",
            "confidence": "high",
            "tone": "defensive",
            "reasoning": "Direct negative outcome ('declined'), significant impact (200 bps), defensive framing ('working on'). Future improvement mentioned but vague.",
            "score": -0.6,
            "key_phrases": ["declined", "increased costs", "working on improvements"]
        }
    }

    SENTIMENT_EXAMPLE_MIXED = {
        "context": "While revenue growth was strong at 12%, we faced some headwinds in our international markets which impacted overall profitability.",
        "analysis": {
            "sentiment_type": "mixed",
            "confidence": "high",
            "tone": "balanced",
            "reasoning": "Clear positive (strong revenue growth, 12%) balanced with clear negative (headwinds, impacted profitability). Both aspects explicitly stated.",
            "score": 0.2,
            "key_phrases": ["strong growth", "headwinds", "impacted profitability"]
        }
    }
