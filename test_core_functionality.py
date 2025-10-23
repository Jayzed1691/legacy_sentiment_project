#!/usr/bin/env python3
"""
Test core functionality: preprocessing and entity recognition
"""

import sys
sys.path.insert(0, 'src')

from legacy_sentiment.processing.preprocessing import TextPreprocessor, PreprocessingConfig
from legacy_sentiment.processing.EntityMWEHandler import EntityMWEHandler

print("=" * 70)
print("LEGACY SENTIMENT PROJECT - CORE FUNCTIONALITY TEST")
print("=" * 70)

# Test 1: Preprocessing Pipeline
print("\n📝 TEST 1: Preprocessing Pipeline")
print("-" * 70)

try:
    config = PreprocessingConfig.from_file('preprocessing_config.json')
    preprocessor = TextPreprocessor.from_config(config)

    test_text = """
    Our Q3 revenue increased by approximately 15% year-over-year to $2.5 billion.
    EBITDA margins expanded to 28%, driven by operational efficiency improvements.
    We expect continued growth throughout fiscal 2024.
    """

    result = preprocessor.preprocess(test_text)

    print(f"✓ Preprocessing successful")
    print(f"  - Tokens extracted: {len(result.get('tokens', []))}")
    print(f"  - Patterns found: {len(result.get('custom_patterns', []))}")
    print(f"  - Cleaned text length: {len(result.get('cleaned_text', ''))}")

    if result.get('custom_patterns'):
        print(f"\n  Detected patterns:")
        for pattern in result['custom_patterns'][:5]:  # Show first 5
            print(f"    - {pattern}")

except Exception as e:
    print(f"✗ Preprocessing failed: {e}")

# Test 2: Entity Recognition
print("\n\n🏢 TEST 2: Entity Recognition")
print("-" * 70)

try:
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

    # Show active components
    components = handler.get_active_components()
    print(f"Active components:")
    for comp, status in components.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {comp}: {status}")

    # Test text with financial entities
    test_text = """
    Apple Inc. reported strong earnings for Q3 2024, with revenue of $95.5 billion.
    The CFO noted that approximately 40% came from iPhone sales.
    Year-over-year growth was 8.5%, exceeding analyst expectations.
    """

    print(f"\n📄 Test text:\n{test_text}")

    results = handler.process_text(test_text)

    print(f"\n📊 Results:")
    total_entities = 0
    entity_sources = {}

    for sentence_num, (sentence_label, (entities, tokens, cleaned)) in enumerate(results, 1):
        print(f"\nSentence {sentence_num}:")

        if entities:
            print(f"  Entities found: {len(entities)}")
            for entity, label, source, start, end in entities:
                entity_sources[source] = entity_sources.get(source, 0) + 1
                total_entities += 1
                print(f"    - '{entity}' | Type: {label} | Source: {source}")
        else:
            print(f"  No entities found")

        print(f"  Tokens processed: {len(tokens)}")
        print(f"  Cleaned text: {cleaned[:80]}...")

    print(f"\n📈 Summary:")
    print(f"  Total entities: {total_entities}")
    print(f"  By source: {entity_sources}")

    if total_entities > 0:
        print(f"\n✓ Entity recognition working correctly")
    else:
        print(f"\n⚠️  No entities detected - check dictionaries")

except Exception as e:
    print(f"✗ Entity recognition failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("CORE FUNCTIONALITY TEST COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("1. If tests passed: System is functional, proceed to evaluate sentiment analysis needs")
print("2. If tests failed: Debug specific failures above")
print("3. Review entity detection quality and adjust dictionaries as needed")
print()
