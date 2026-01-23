#!/usr/bin/env python3
"""
Confidence Calibration Experiment for Voice Assistant Context Resolution

PURPOSE:
--------
Validate that the 70/90 confidence thresholds are empirically grounded.
If queries with 90%+ confidence are only correct 70% of the time, we need to recalibrate.

METHODOLOGY:
------------
1. Run diverse test queries through context resolution
2. Compare model's confidence to human-judged correctness
3. Calculate Expected Calibration Error (ECE)
4. Adjust thresholds if needed

USAGE:
------
    python calibration_experiment.py --run        # Run full experiment
    python calibration_experiment.py --analyze    # Analyze existing results
    python calibration_experiment.py --interactive  # Interactive labeling

Based on: "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017)
         "Just Ask for Calibration" (Tian et al., EMNLP 2023)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse
import sys

# Import the router we're testing
try:
    from search_router import init_router, _phase1_resolve_context, TopicCache
    from groq import Groq
    import os
except ImportError as e:
    print(f"Import error: {e}")
    print("Run from the cpu-test directory with venv activated")
    sys.exit(1)


@dataclass
class CalibrationSample:
    """A single calibration test case."""
    query: str
    context_setup: str  # Previous conversation context
    expected_needs_context: bool  # Human judgment: does this need history?
    expected_topic: Optional[str] = None  # What topic should be extracted
    expected_reformulation: Optional[str] = None  # Expected standalone form

    # Results (filled after testing)
    model_confidence: Optional[float] = None
    model_reformulation: Optional[str] = None
    model_topic: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_ms: Optional[int] = None


@dataclass
class CalibrationResults:
    """Aggregated calibration results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_samples: int = 0

    # Calibration by confidence bucket
    bucket_90_100: dict = field(default_factory=lambda: {"total": 0, "correct": 0})
    bucket_70_89: dict = field(default_factory=lambda: {"total": 0, "correct": 0})
    bucket_40_69: dict = field(default_factory=lambda: {"total": 0, "correct": 0})
    bucket_0_39: dict = field(default_factory=lambda: {"total": 0, "correct": 0})

    # Expected Calibration Error
    ece: Optional[float] = None

    # Recommended thresholds (may differ from 70/90)
    recommended_high_threshold: Optional[float] = None
    recommended_low_threshold: Optional[float] = None


# ============================================================================
# TEST DATASET
# ============================================================================

# Diverse test cases covering different domains and linguistic patterns
# Format: (context_setup, query, expected_needs_context, expected_topic)

TEST_DATASET = [
    # --- Pronouns (should need context) ---
    (
        "Tell me about Tesla's latest earnings",
        "What's their stock price?",
        True,
        "Tesla"
    ),
    (
        "I'm learning Python programming",
        "Can you give me another example?",
        True,
        "Python"
    ),
    (
        "The Lakers played the Celtics last night",
        "Who won?",
        True,
        "Lakers vs Celtics game"
    ),
    (
        "My car is a Toyota Camry",
        "How much does it cost to service it?",
        True,
        "Toyota Camry"
    ),
    (
        "I'm reading about quantum computing",
        "How does it work?",
        True,
        "quantum computing"
    ),

    # --- Demonstratives (should need context) ---
    (
        "You mentioned recursion earlier",
        "That sounds complicated",
        True,
        "recursion"
    ),
    (
        "The restaurant got 4 stars on Yelp",
        "Is that good?",
        True,
        "restaurant"
    ),
    (
        "We discussed the new tax law",
        "This affects everyone?",
        True,
        "new tax law"
    ),
    (
        "Einstein developed special relativity",
        "Those equations are famous",
        True,
        "special relativity"
    ),

    # --- Ellipsis (missing subject - should need context) ---
    (
        "I want to buy a new laptop",
        "And the price?",
        True,
        "laptop"
    ),
    (
        "We're planning a trip to Japan",
        "Best time to visit?",
        True,
        "Japan trip"
    ),
    (
        "The iPhone 16 was just announced",
        "Release date?",
        True,
        "iPhone 16"
    ),
    (
        "My doctor recommended exercise",
        "How often?",
        True,
        "exercise"
    ),

    # --- Continuations (should need context) ---
    (
        "Bitcoin hit $50,000 today",
        "What about Ethereum?",
        True,
        "cryptocurrency prices"
    ),
    (
        "The weather in New York is cold",
        "Same in Chicago?",
        True,
        "weather"
    ),
    (
        "I tried the new Italian restaurant",
        "Did you like the pasta?",
        True,
        "Italian restaurant"
    ),

    # --- Topic Shift (should NOT need context) ---
    (
        "Tell me about Tesla's latest earnings",
        "What's the capital of France?",
        False,
        None  # New topic, no context needed
    ),
    (
        "I'm learning Python programming",
        "How's the weather today?",
        False,
        None
    ),
    (
        "The Lakers played last night",
        "Who invented the telephone?",
        False,
        None
    ),
    (
        "We discussed the tax law",
        "What time is it?",
        False,
        None
    ),

    # --- Standalone Questions (no context present) ---
    (
        "",  # No prior context
        "What's the population of Tokyo?",
        False,
        None
    ),
    (
        "",
        "How do I make pancakes?",
        False,
        None
    ),
    (
        "",
        "What's the meaning of life?",
        False,
        None
    ),
    (
        "",
        "Tell me a joke",
        False,
        None
    ),

    # --- Ambiguous Cases (edge cases) ---
    (
        "I love pizza",
        "Where can I get some?",
        True,  # "some" refers to pizza
        "pizza"
    ),
    (
        "The meeting is at 3pm",
        "Can we reschedule?",
        True,  # "we" implies shared context
        "meeting"
    ),
    (
        "I'm thinking about moving",
        "Is it expensive?",
        True,  # "it" refers to moving
        "moving"
    ),

    # --- False Positives Risk (pronouns that don't need context) ---
    (
        "",
        "It is raining outside",
        False,  # "It" is weather impersonal, not referential
        None
    ),
    (
        "",
        "They say breakfast is important",
        False,  # Generic "they"
        None
    ),
    (
        "Hello!",
        "That's a nice day",
        False,  # Not referring to prior topic
        None
    ),

    # --- Complex Multi-Turn (deep context) ---
    (
        "User: Tell me about SpaceX\nAssistant: SpaceX is Elon Musk's rocket company...\nUser: What about their rockets?",
        "How many have they launched?",
        True,
        "SpaceX rockets"
    ),
    (
        "User: I have a headache\nAssistant: Have you tried ibuprofen?\nUser: Yes, twice today",
        "Should I take more?",
        True,
        "ibuprofen dosage"
    ),

    # --- Domain-Specific Examples (testing generalization) ---
    # Sports
    (
        "The Chiefs are playing the Eagles",
        "What's the spread?",
        True,
        "Chiefs vs Eagles"
    ),
    # Finance
    (
        "Apple's market cap is huge",
        "Bigger than Microsoft?",
        True,
        "Apple vs Microsoft market cap"
    ),
    # Technology
    (
        "GPT-5 was just released",
        "What's different about it?",
        True,
        "GPT-5"
    ),
    # Healthcare
    (
        "My blood pressure is 140/90",
        "Is that high?",
        True,
        "blood pressure"
    ),
    # Cooking
    (
        "I'm making spaghetti carbonara",
        "How long do I cook the pasta?",
        True,
        "spaghetti carbonara"
    ),
    # Travel
    (
        "Planning a trip to Paris",
        "Do I need a visa?",
        True,
        "Paris trip visa"
    ),
    # Education
    (
        "I'm studying calculus",
        "Can you explain derivatives?",
        True,
        "calculus derivatives"
    ),

    # --- Edge: Very Short Queries ---
    (
        "We're talking about dogs",
        "Cute!",
        False,  # Exclamation, not a question needing context
        None
    ),
    (
        "I mentioned the meeting",
        "When?",
        True,  # Single word but clearly refers to meeting
        "meeting"
    ),

    # --- Explicit Topic (should extract even without pronouns) ---
    (
        "The election results are in",
        "What happened with the electoral college?",
        True,  # Related to election context
        "election results"
    ),
    (
        "I love my new car",
        "The maintenance schedule is important",
        True,  # Definite "the" refers to the car
        "car maintenance"
    ),
]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class CalibrationExperiment:
    """Runs calibration experiment on the search router."""

    def __init__(self, groq_client):
        self.groq_client = groq_client
        init_router(groq_client)
        self.results_dir = Path(__file__).parent / "calibration_results"
        self.results_dir.mkdir(exist_ok=True)

    def build_history_from_context(self, context_setup: str) -> list:
        """Convert context string to conversation history format."""
        if not context_setup:
            return []

        # Parse multi-turn format if present
        if "User:" in context_setup and "Assistant:" in context_setup:
            history = []
            lines = context_setup.strip().split("\n")
            current_user = None
            current_assistant = None

            for line in lines:
                if line.startswith("User: "):
                    if current_user and current_assistant:
                        history.append({"user": current_user, "assistant": current_assistant})
                    current_user = line[6:]
                    current_assistant = None
                elif line.startswith("Assistant: "):
                    current_assistant = line[11:]

            if current_user and current_assistant:
                history.append({"user": current_user, "assistant": current_assistant})
            return history

        # Single statement becomes a user/assistant pair
        return [{"user": context_setup, "assistant": "I understand. Let me help with that."}]

    def run_single_test(self, sample: CalibrationSample) -> CalibrationSample:
        """Run a single test case through the router."""
        history = self.build_history_from_context(sample.context_setup)

        start = time.time()
        resolved, confidence, topic = _phase1_resolve_context(sample.query, history)
        latency = int((time.time() - start) * 1000)

        # Fill in results
        sample.model_confidence = confidence
        sample.model_reformulation = resolved
        sample.model_topic = topic
        sample.latency_ms = latency

        # Determine correctness
        # Correct if: model's context-need assessment matches expected
        model_thinks_needs_context = confidence < 0.5 or (resolved != sample.query)
        sample.is_correct = model_thinks_needs_context == sample.expected_needs_context

        return sample

    def run_experiment(self, dataset: list = None) -> CalibrationResults:
        """Run the full calibration experiment."""
        dataset = dataset or TEST_DATASET
        samples = []

        print(f"\nRunning calibration experiment on {len(dataset)} samples...")
        print("-" * 60)

        for i, (context, query, needs_context, expected_topic) in enumerate(dataset):
            sample = CalibrationSample(
                query=query,
                context_setup=context,
                expected_needs_context=needs_context,
                expected_topic=expected_topic
            )

            result = self.run_single_test(sample)
            samples.append(result)

            status = "correct" if result.is_correct else "WRONG"
            print(f"[{i+1:3d}/{len(dataset)}] conf={result.model_confidence:.2f} "
                  f"{status:7s} | '{query[:40]}...' → '{result.model_reformulation[:40]}...'")

        # Aggregate results
        results = self._aggregate_results(samples)

        # Save results
        self._save_results(samples, results)

        return results

    def _aggregate_results(self, samples: list) -> CalibrationResults:
        """Calculate calibration metrics from samples."""
        results = CalibrationResults(total_samples=len(samples))

        # Bucket samples by confidence
        for sample in samples:
            conf = sample.model_confidence or 0
            correct = 1 if sample.is_correct else 0

            if conf >= 0.9:
                results.bucket_90_100["total"] += 1
                results.bucket_90_100["correct"] += correct
            elif conf >= 0.7:
                results.bucket_70_89["total"] += 1
                results.bucket_70_89["correct"] += correct
            elif conf >= 0.4:
                results.bucket_40_69["total"] += 1
                results.bucket_40_69["correct"] += correct
            else:
                results.bucket_0_39["total"] += 1
                results.bucket_0_39["correct"] += correct

        # Calculate Expected Calibration Error (ECE)
        # ECE = sum( |accuracy(bucket) - avg_confidence(bucket)| * n_bucket / n_total )
        buckets = [
            (results.bucket_90_100, 0.95),  # midpoint confidence
            (results.bucket_70_89, 0.795),
            (results.bucket_40_69, 0.545),
            (results.bucket_0_39, 0.195),
        ]

        ece = 0.0
        for bucket, avg_conf in buckets:
            if bucket["total"] > 0:
                accuracy = bucket["correct"] / bucket["total"]
                ece += abs(accuracy - avg_conf) * bucket["total"] / len(samples)

        results.ece = ece

        # Recommend new thresholds if calibration is off
        # If 90%+ bucket has < 85% accuracy, raise the high threshold
        if results.bucket_90_100["total"] > 0:
            acc_90 = results.bucket_90_100["correct"] / results.bucket_90_100["total"]
            if acc_90 < 0.85:
                results.recommended_high_threshold = 0.95
            else:
                results.recommended_high_threshold = 0.90

        if results.bucket_70_89["total"] > 0:
            acc_70 = results.bucket_70_89["correct"] / results.bucket_70_89["total"]
            if acc_70 < 0.65:
                results.recommended_low_threshold = 0.75
            else:
                results.recommended_low_threshold = 0.70

        return results

    def _save_results(self, samples: list, results: CalibrationResults):
        """Save results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save samples
        samples_file = self.results_dir / f"samples_{timestamp}.json"
        with open(samples_file, "w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)

        # Save aggregated results
        results_file = self.results_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(asdict(results), f, indent=2)

        print(f"\nResults saved to:")
        print(f"  - {samples_file}")
        print(f"  - {results_file}")

    def print_report(self, results: CalibrationResults):
        """Print a human-readable calibration report."""
        print("\n" + "=" * 60)
        print("CALIBRATION REPORT")
        print("=" * 60)
        print(f"\nTotal samples: {results.total_samples}")
        print(f"Timestamp: {results.timestamp}")

        print("\nConfidence Bucket Analysis:")
        print("-" * 40)

        buckets = [
            ("90-100%", results.bucket_90_100),
            ("70-89%", results.bucket_70_89),
            ("40-69%", results.bucket_40_69),
            ("0-39%", results.bucket_0_39),
        ]

        for name, bucket in buckets:
            if bucket["total"] > 0:
                accuracy = bucket["correct"] / bucket["total"] * 100
                print(f"  {name:10s}: {bucket['correct']:3d}/{bucket['total']:3d} correct ({accuracy:.1f}%)")
            else:
                print(f"  {name:10s}: No samples")

        print(f"\nExpected Calibration Error (ECE): {results.ece:.3f}")
        print(f"  (Lower is better, 0.0 = perfect calibration)")

        print("\nThreshold Recommendations:")
        if results.recommended_high_threshold:
            print(f"  High confidence threshold: {results.recommended_high_threshold:.2f} "
                  f"(current: 0.90)")
        if results.recommended_low_threshold:
            print(f"  Low confidence threshold: {results.recommended_low_threshold:.2f} "
                  f"(current: 0.70)")

        # Assessment
        print("\nAssessment:")
        if results.ece < 0.1:
            print("  model is well-calibrated for production use")
        elif results.ece < 0.2:
            print("  Model is reasonably calibrated, minor adjustments recommended")
        else:
            print("  WARNING: Model is poorly calibrated, threshold adjustment required!")

        print("=" * 60)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibration experiment for voice assistant context resolution"
    )
    parser.add_argument("--run", action="store_true", help="Run full experiment")
    parser.add_argument("--analyze", type=str, help="Analyze existing results file")
    parser.add_argument("--interactive", action="store_true", help="Interactive labeling mode")
    parser.add_argument("--quick", action="store_true", help="Run quick test (10 samples)")

    args = parser.parse_args()

    if not any([args.run, args.analyze, args.interactive, args.quick]):
        parser.print_help()
        print("\nExample:")
        print("  python calibration_experiment.py --run")
        print("  python calibration_experiment.py --quick")
        return

    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY environment variable not set")
        print("Set it with: export GROQ_API_KEY='your-key'")
        return

    groq_client = Groq(api_key=groq_api_key)
    experiment = CalibrationExperiment(groq_client)

    if args.run or args.quick:
        dataset = TEST_DATASET[:10] if args.quick else TEST_DATASET
        results = experiment.run_experiment(dataset)
        experiment.print_report(results)

    elif args.analyze:
        # Load and analyze existing results
        with open(args.analyze) as f:
            data = json.load(f)
        results = CalibrationResults(**data)
        experiment.print_report(results)

    elif args.interactive:
        print("Interactive labeling mode not yet implemented")
        print("Use --run to run the experiment with pre-labeled data")


if __name__ == "__main__":
    main()
