#!/usr/bin/env python3
"""
Test Refactored Transcript Parser

Quick test to verify the new sentiment_analyzer.core.transcript module works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentiment_analyzer.core.transcript import parse_transcript, format_transcript


def test_json_parser():
    """Test JSON transcript parsing."""
    print("=" * 70)
    print("Testing JSON Transcript Parser")
    print("=" * 70)

    transcript = parse_transcript('data/transcripts/earnings_call_sample.json', 'json')

    print(f"\n✓ Parsed JSON transcript")
    print(f"  Sections: {len(transcript.sections)}")
    print(f"  Speakers: {len(transcript.speakers)}")
    print(f"  Total dialogues: {len(transcript.all_dialogues())}")

    # Show first section
    if transcript.sections:
        section = transcript.sections[0]
        print(f"\n  First section: {section.name}")
        print(f"    Dialogues: {len(section.dialogues)}")

        if section.dialogues:
            dialogue = section.dialogues[0]
            print(f"\n    First speaker: {dialogue.speaker} ({dialogue.role})")
            print(f"    Text preview: {dialogue.text[:100]}...")

    # Show speaker breakdown
    print(f"\n  Speaker breakdown:")
    for speaker, texts in list(transcript.speakers.items())[:3]:
        print(f"    {speaker}: {len(texts)} dialogue(s)")


def test_txt_parser():
    """Test TXT transcript parsing."""
    print("\n" + "=" * 70)
    print("Testing TXT Transcript Parser")
    print("=" * 70)

    transcript = parse_transcript('data/transcripts/earnings_call_sample.txt', 'txt')

    print(f"\n✓ Parsed TXT transcript")
    print(f"  Sections: {len(transcript.sections)}")
    print(f"  Speakers: {len(transcript.speakers)}")
    print(f"  Total dialogues: {len(transcript.all_dialogues())}")

    # Show first section
    if transcript.sections:
        section = transcript.sections[0]
        print(f"\n  First section: {section.name}")
        print(f"    Dialogues: {len(section.dialogues)}")

        if section.dialogues:
            dialogue = section.dialogues[0]
            print(f"\n    First speaker: {dialogue.speaker} ({dialogue.role})")
            print(f"    Text preview: {dialogue.text[:100]}...")


def test_formatting():
    """Test transcript formatting."""
    print("\n" + "=" * 70)
    print("Testing Transcript Formatting")
    print("=" * 70)

    transcript = parse_transcript('data/transcripts/earnings_call_sample.json', 'json')
    formatted = format_transcript(transcript)

    print(f"\n✓ Formatted transcript ({len(formatted)} characters)")
    print(f"\n  Preview (first 500 chars):")
    print("-" * 70)
    print(formatted[:500])
    print("-" * 70)


def test_speaker_queries():
    """Test speaker-specific queries."""
    print("\n" + "=" * 70)
    print("Testing Speaker Queries")
    print("=" * 70)

    transcript = parse_transcript('data/transcripts/earnings_call_sample.json', 'json')

    # Get all speakers
    speakers = list(transcript.speakers.keys())
    print(f"\n✓ Found {len(speakers)} speakers:")
    for speaker in speakers[:5]:  # Show first 5
        dialogues = transcript.get_speaker_dialogues(speaker)
        print(f"  {speaker}: {len(dialogues)} dialogue(s)")


if __name__ == "__main__":
    try:
        test_json_parser()
        test_txt_parser()
        test_formatting()
        test_speaker_queries()

        print("\n" + "=" * 70)
        print("✅ All Tests Passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
