"""
Quick diagnostic script to investigate why validation accuracy is stuck at 27.82%.
Checks: label distribution, timestamp parsing, and whether the model is collapsing
to a single class prediction.
"""
import os
import json
import sys
from collections import Counter

# Add parent to path
sys.path.insert(0, r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_2")

from SoccerNet.utils import getListGames

ROOT_PATH = r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_2\SoccerNet_Dataset"

label_to_id = {
    "Background": 0, "Ball out of play": 1, "Clearance": 2, "Corner": 3,
    "Direct free-kick": 4, "Foul": 5, "Goal": 6, "Indirect free-kick": 7,
    "Injury": 8, "Kick-off": 9, "Offside": 10, "Penalty": 11,
    "Red card": 12, "Shots off target": 13, "Shots on target": 14,
    "Substitution": 15, "Throw-in": 16, "Yellow card": 17
}

for split in ["train", "valid"]:
    games = getListGames(split=split)
    label_counts = Counter()
    timestamp_errors = 0
    total = 0
    
    for game in games:
        game_path = os.path.join(ROOT_PATH, game)
        json_path = os.path.join(game_path, "Labels-v2.json")
        video_path = os.path.join(game_path, "1_224p.mkv")
        
        if os.path.exists(json_path) and os.path.exists(video_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                for annotation in data['annotations']:
                    if annotation['label'] in label_to_id:
                        label_counts[annotation['label']] += 1
                        total += 1
                        
                        # Check timestamp parsing
                        try:
                            parts = annotation['gameTime'].split(' - ')
                            timestamp = parts if len(parts) > 1 else parts
                            minutes, seconds = map(float, timestamp.split(':'))
                        except:
                            timestamp_errors += 1

    print(f"\n{'='*60}")
    print(f"SPLIT: {split.upper()} | Total samples: {total}")
    print(f"{'='*60}")
    print(f"Timestamp parsing errors: {timestamp_errors}/{total}")
    
    # Sort by count descending
    print(f"\n{'Label':<25} {'Count':>6} {'Pct':>8}")
    print("-" * 42)
    for label, count in label_counts.most_common():
        pct = (count / total) * 100
        print(f"{label:<25} {count:>6} {pct:>7.2f}%")
    
    # Check if any single class == 27.82%
    print(f"\n--- Checking for 27.82% match ---")
    for label, count in label_counts.most_common():
        pct = (count / total) * 100
        if abs(pct - 27.82) < 0.5:
            print(f"  MATCH: '{label}' = {pct:.2f}% ({count}/{total})")

    # What accuracy would you get predicting the most common class?
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    majority_pct = (most_common_count / total) * 100
    print(f"\n  Majority class: '{most_common_label}' ({majority_pct:.2f}%)")
    print(f"  If model predicts ONLY '{most_common_label}', accuracy = {majority_pct:.2f}%")
