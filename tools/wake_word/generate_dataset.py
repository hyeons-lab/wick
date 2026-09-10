#!/usr/bin/env python3
"""
Automated dataset prompt and phonetic near-miss generator for generic KWS training.

Generates positive variants, algorithmic phonetic near-misses (rhymes, substitutions,
collocations), and negative confusers for any target keyword phrase.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Set, Tuple


# ── Built-in Phonetic Dictionary & Substitution Rules ─────────────────────────

# Common English phonetic substitutions and near-rhyme sets for wake words
COMMON_RHYMES_AND_NEAR_MISSES: Dict[str, List[str]] = {
    "liquid": ["squid", "livid", "lipid", "lizard", "limit", "rigid", "frigid", "vivid", "wicked"],
    "hey": ["pay", "play", "they", "say", "may", "day", "way", "gray", "ray", "bay", "nay"],
    "hi": ["pie", "fly", "my", "high", "sigh", "tie", "why", "lie", "buy"],
    "okay": ["decay", "today", "relay", "delay", "display", "replay"],
    "computer": ["commuter", "polluter", "recruiter", "disputer", "shooter", "scooter"],
    "jarvis": ["harvest", "tardis", "carcass", "artist", "service"],
    "alexa": ["aretha", "alaska", "amnesia", "alyssa"],
    "siri": ["cereal", "cheerie", "weary", "eerie", "theory"],
    "google": ["giggle", "boggle", "gargle", "noodle", "doodle"],
}

COMMON_COLLOCATIONS: Dict[str, List[str]] = {
    "liquid": [
        "liquid soap",
        "liquid nitrogen",
        "liquid paper",
        "liquid detergent",
        "dishwashing liquid",
        "spilled some liquid",
        "clear liquid",
    ],
    "computer": [
        "computer screen",
        "computer mouse",
        "computer science",
        "my computer crashed",
        "turn on computer",
    ],
}

CONFUSER_WAKE_WORDS = [
    "Hey Siri",
    "Hey Google",
    "Okay Google",
    "Alexa",
    "Computer",
    "Hey Cortana",
    "Jarvis",
]


# ── Near-Miss & Variant Mining ────────────────────────────────────────────────


def generate_phonetic_near_misses(keyword: str) -> Set[str]:
    """
    Generate phonetically confusable phrases using rhyme and substitution rules.
    """
    words = keyword.lower().strip().split()
    near_misses: Set[str] = set()

    for idx, word in enumerate(words):
        rhymes = COMMON_RHYMES_AND_NEAR_MISSES.get(word, [])
        for rhyme in rhymes:
            # Replace target word with rhyme
            variant_words = list(words)
            variant_words[idx] = rhyme
            near_misses.add(" ".join(variant_words))

        # Add single-word near-misses
        for rhyme in rhymes:
            near_misses.add(rhyme)

    # Word boundary confusions (e.g. "They liquid", "Play liquid" for "Hey liquid")
    if len(words) >= 2 and words[0] in COMMON_RHYMES_AND_NEAR_MISSES:
        first_rhymes = COMMON_RHYMES_AND_NEAR_MISSES[words[0]]
        remainder = " ".join(words[1:])
        for r in first_rhymes:
            near_misses.add(f"{r} {remainder}")

    # Add collocations for constituent words
    for word in words:
        if word in COMMON_COLLOCATIONS:
            for collocation in COMMON_COLLOCATIONS[word]:
                near_misses.add(collocation)

    # Filter out exact keyword
    near_misses.discard(keyword.lower().strip())
    return near_misses


def generate_positive_variants(keyword: str, aliases: List[str]) -> List[Dict[str, any]]:
    """
    Generate positive utterance definitions with acoustic perturbation parameters.
    """
    phrasings = [keyword] + aliases
    # Add comma-pause variants (e.g. "Hey, Liquid")
    comma_variants = []
    for p in phrasings:
        parts = p.split()
        if len(parts) >= 2:
            comma_variants.append(f"{parts[0]}, {' '.join(parts[1:])}")
    phrasings.extend(comma_variants)

    pitch_shifts = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
    speaking_rates = [0.8, 0.9, 1.0, 1.1, 1.2]
    snr_levels = [None, 20.0, 15.0, 10.0]  # dB SNR for noise mixing

    positives = []
    for text in phrasings:
        for pitch in pitch_shifts:
            for rate in speaking_rates:
                for snr in snr_levels:
                    positives.append({
                        "text": text,
                        "pitch_shift": pitch,
                        "speaking_rate": rate,
                        "target_snr_db": snr,
                        "is_positive": True,
                    })

    return positives


def generate_negative_variants(keyword: str, count: int = 5000) -> List[Dict[str, any]]:
    """
    Generate negative sample definitions covering near-rhymes, confusers, and chatter.
    """
    near_misses = sorted(list(generate_phonetic_near_misses(keyword)))
    all_negatives = list(near_misses) + CONFUSER_WAKE_WORDS

    samples = []
    # Guarantee all near misses and confusers are included
    for text in all_negatives:
        samples.append({
            "text": text,
            "category": "hard_negative",
            "is_positive": False,
        })

    # Expand with acoustic variations
    pitch_shifts = [0.90, 1.0, 1.10]
    speaking_rates = [0.9, 1.0, 1.1]

    while len(samples) < count:
        base_text = random.choice(all_negatives)
        samples.append({
            "text": base_text,
            "pitch_shift": random.choice(pitch_shifts),
            "speaking_rate": random.choice(speaking_rates),
            "category": "hard_negative_perturbed",
            "is_positive": False,
        })

    return samples[:count]


# ── Main Entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training prompt dataset and phonetic near-misses for KWS"
    )
    parser.add_argument(
        "--keyword",
        default="Hey Liquid",
        help="Primary target keyword (e.g. 'Hey Liquid')",
    )
    parser.add_argument(
        "--aliases",
        default="Okay Liquid,Hi Liquid",
        help="Comma-separated secondary aliases",
    )
    parser.add_argument(
        "--output",
        default="tools/wake_word/dataset_manifest.json",
        help="Path to output JSON dataset manifest",
    )
    parser.add_argument(
        "--neg-count",
        type=int,
        default=2000,
        help="Number of negative prompt variations to generate (default: 2000)",
    )
    args = parser.parse_args()

    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]

    print(f"Generating positive prompt variations for '{args.keyword}'...")
    positives = generate_positive_variants(args.keyword, aliases)
    print(f"Generated {len(positives):,} positive sample configurations.")

    print(f"Mining phonetic near-misses and hard negatives for '{args.keyword}'...")
    near_misses = generate_phonetic_near_misses(args.keyword)
    print(f"Mined {len(near_misses)} unique phonetic near-misses: {sorted(list(near_misses))}")

    negatives = generate_negative_variants(args.keyword, count=args.neg_count)
    print(f"Generated {len(negatives):,} negative sample configurations.")

    manifest = {
        "keyword": args.keyword,
        "aliases": aliases,
        "summary": {
            "total_positives": len(positives),
            "total_negatives": len(negatives),
            "hard_near_misses": sorted(list(near_misses)),
        },
        "positives": positives,
        "negatives": negatives,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully wrote dataset manifest to {args.output}")


if __name__ == "__main__":
    main()
