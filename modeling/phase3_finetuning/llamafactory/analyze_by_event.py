#!/usr/bin/env python3
"""
Analyze model performance by event type
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def extract_event_type(item):
    """Extract event type from input text"""
    input_text = item.get("input", "")
    
    # Look for "Event Type: X.XX" pattern
    for line in input_text.split("\n"):
        if "Event Type:" in line:
            # Extract the item number (e.g., "7.01", "9.01")
            parts = line.split("Event Type:")[1].strip().split("-")[0].strip()
            return parts
    return "Unknown"

def analyze_by_event(results_path, data_path):
    """Analyze results by event type"""
    
    # Load evaluation results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load validation data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predictions = results["predictions"]
    true_labels = results["true_labels"]
    
    # Group by event type
    event_results = defaultdict(lambda: {"predictions": [], "true_labels": []})
    
    for i, item in enumerate(data):
        event_type = extract_event_type(item)
        event_results[event_type]["predictions"].append(predictions[i])
        event_results[event_type]["true_labels"].append(true_labels[i])
    
    # Calculate metrics per event type
    print("\n" + "="*70)
    print("PERFORMANCE BY EVENT TYPE")
    print("="*70)
    
    event_metrics = {}
    
    for event_type in sorted(event_results.keys()):
        preds = event_results[event_type]["predictions"]
        labels = event_results[event_type]["true_labels"]
        
        # Calculate accuracy
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        accuracy = correct / len(preds) if preds else 0
        
        # Count samples
        num_samples = len(preds)
        
        # Store metrics
        event_metrics[event_type] = {
            "accuracy": accuracy,
            "num_samples": num_samples,
            "correct": correct
        }
        
        # Print results
        print(f"\nEvent Type {event_type}:")
        print(f"  Samples: {num_samples}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{num_samples})")
        
        # Distribution of predictions
        pred_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for p in preds:
            pred_dist[p] += 1
        print(f"  Predictions: S_NEG={pred_dist[0]}, NEG={pred_dist[1]}, POS={pred_dist[2]}, S_POS={pred_dist[3]}")
    
    # Overall statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Sort by accuracy
    sorted_events = sorted(event_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print("\nTop Performing Event Types:")
    for i, (event_type, metrics) in enumerate(sorted_events[:5], 1):
        print(f"  {i}. Event {event_type}: {metrics['accuracy']:.4f} accuracy ({metrics['num_samples']} samples)")
    
    print("\nBottom Performing Event Types:")
    for i, (event_type, metrics) in enumerate(sorted_events[-5:], 1):
        print(f"  {i}. Event {event_type}: {metrics['accuracy']:.4f} accuracy ({metrics['num_samples']} samples)")
    
    # Save detailed results
    output_path = "event_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(event_metrics, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")
    
    return event_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="evaluation_results.json")
    parser.add_argument("--data_path", type=str, default="data/val.json")
    args = parser.parse_args()
    
    analyze_by_event(args.results_path, args.data_path)

if __name__ == "__main__":
    main()