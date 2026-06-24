import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from SoccerNet.utils import getListGames
import SoccerNet.Evaluation.ActionSpotting as AS
from SoccerNet.Evaluation.ActionSpotting import average_mAP, label2vector, predictions2vector, EVENT_DICTIONARY_V2

# 1. PATHS & CONFIGURATION
# Dynamic paths for cluster deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = os.getenv("SOCCERNET_DIR", os.path.join(BASE_DIR, "SoccerNet_Dataset"))
PREDS_PATH = os.path.join(BASE_DIR, "evaluation", "predictions")
STRIDE = 0.5         # Stride/step size in seconds for sliding window
WINDOW_SIZE = 10.0    # Window size in seconds for peak detection
CONF_THRESHOLD = 0.4  # Minimum confidence to consider a spot
BATCH_SIZE = 32

# ── CHECKPOINT SELECTION ────────────────────────────────────────────────────
# To evaluate a specific run, set CHECKPOINT_PATH explicitly.
# Set to None to auto-detect the latest .pth in exp2d_unfrozen_blocks4_5.
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "exp2d_unfrozen_blocks4_5", "unfrozen_blocks45_bs2_lr5e-05_focal_loss_epoch_24.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_labels = {
    0: "Background", 1: "Ball out of play", 2: "Clearance", 3: "Corner",
    4: "Direct free-kick", 5: "Foul", 6: "Goal", 7: "Indirect free-kick",
    8: "Injury", 9: "Kick-off", 10: "Offside", 11: "Penalty",
    12: "Red card", 13: "Shots off target", 14: "Shots on target",
    15: "Substitution", 16: "Throw-in", 17: "Yellow card"
}

def get_model(num_classes=18):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features=in_features, out_features=num_classes)
    return model

def load_latest_weights(model, device):
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Loading specified checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        return CHECKPOINT_PATH
    # Fallback: auto-detect latest in exp2d_unfrozen_blocks4_5
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "exp2d_unfrozen_blocks4_5")
    list_of_weights = glob.glob(os.path.join(ckpt_dir, "*focal_loss*.pth"))
    if not list_of_weights:
        raise FileNotFoundError(f"No trained .pth weights found in {ckpt_dir}")
    latest_weights = max(list_of_weights, key=os.path.getctime)
    print(f"\n[INFO] Auto-detected latest weights: {latest_weights}")
    model.load_state_dict(torch.load(latest_weights, map_location=device))
    return latest_weights

def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[INFO] Running inference on: {video_path}")
    print(f"       Total Frames: {total_frames} | FPS: {fps} | Length: {total_frames/fps:.1f}s")

    # Generate prediction targets
    target_clips = []
    k = 0
    while True:
        T_k = 1.0 + k * STRIDE
        F_k = int((T_k - 1.0) * fps)
        E_k = F_k + 31
        if E_k >= total_frames:
            break
        target_clips.append((k, T_k, F_k, E_k))
        k += 1

    # Map end_frame index to list of (k, T_k, F_k)
    end_frame_map = {}
    for k, T_k, F_k, E_k in target_clips:
        if E_k not in end_frame_map:
            end_frame_map[E_k] = []
        end_frame_map[E_k].append((k, T_k, F_k))

    frame_buffer = []
    predictions_raw = {}
    
    batch_clips = []
    batch_times = []

    def run_batch():
        if not batch_clips:
            return
        fast_tensors = torch.stack(batch_clips, dim=0).to(device)
        slow_tensors = fast_tensors[:, :, ::4, :, :].to(device)
        with torch.no_grad():
            outputs = model([slow_tensors, fast_tensors])
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        for idx, t in enumerate(batch_times):
            predictions_raw[t] = probs[idx]
        batch_clips.clear()
        batch_times.clear()

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Inferring video frames")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Fast cv2 preprocessing
        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float() / 255.0

        frame_buffer.append(tensor_img)
        if len(frame_buffer) > 32:
            frame_buffer.pop(0)

        if frame_idx in end_frame_map:
            for k, T_k, F_k in end_frame_map[frame_idx]:
                clip = torch.stack(frame_buffer, dim=1)  # [3, 32, 224, 224]
                batch_clips.append(clip)
                batch_times.append(T_k)

                if len(batch_clips) == BATCH_SIZE:
                    run_batch()

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    run_batch()
    return predictions_raw

def perform_peak_detection(predictions_raw):
    times = sorted(predictions_raw.keys())
    if not times:
        return []
    
    num_steps = len(times)
    probs_array = np.zeros((num_steps, 18))
    for idx, t in enumerate(times):
        probs_array[idx] = predictions_raw[t]

    stride_steps = int(WINDOW_SIZE / STRIDE)
    half_window = stride_steps // 2

    detected_spots = []

    # Loop over classes 1 to 17
    for c in range(1, 18):
        class_probs = probs_array[:, c]
        for idx in range(num_steps):
            val = class_probs[idx]
            if val < CONF_THRESHOLD:
                continue
            
            start = max(0, idx - half_window)
            end = min(num_steps, idx + half_window + 1)
            if val == np.max(class_probs[start:end]):
                # Add to spotted list
                t = times[idx]
                detected_spots.append({
                    "time": t,
                    "label": action_labels[c],
                    "confidence": float(val)
                })
    return detected_spots

def main():
    os.makedirs(PREDS_PATH, exist_ok=True)
    
    print(f"Using device: {device}")
    model = get_model(num_classes=18).to(device)
    try:
        weight_file = load_latest_weights(model, device)
    except FileNotFoundError as e:
        print(e)
        return

    model.eval()

    # Get local valid games with 1_224p.mkv video
    all_valid_games = getListGames(split="valid", dataset="SoccerNet", task="spotting")
    local_valid_games = [g for g in all_valid_games if os.path.exists(os.path.join(ROOT_PATH, g, "1_224p.mkv"))]
    
    print(f"\nFound {len(local_valid_games)} validation games locally with video:")
    for g in local_valid_games:
        print(f"  - {g}")

    if not local_valid_games:
        print("No validation videos found. Exiting.")
        return

    # Process each local game
    for game in local_valid_games:
        game_dir = os.path.join(ROOT_PATH, game)
        video_path = os.path.join(game_dir, "1_224p.mkv")
        
        # Run sliding window model inference
        predictions_raw = process_video(video_path, model, device)
        
        # Spot actions with peak detection
        spots = perform_peak_detection(predictions_raw)
        print(f"Detected {len(spots)} candidate action spots.")

        # Save to predictions JSON structure
        pred_list = []
        for spot in spots:
            T = spot["time"]
            minutes = int(T // 60)
            seconds = int(T % 60)
            pred_list.append({
                "gameTime": f"1 - {minutes:02d}:{seconds:02d}",
                "label": spot["label"],
                "position": str(int(T * 1000)),
                "half": "1",
                "confidence": float(spot["confidence"])
            })

        game_pred_dir = os.path.join(PREDS_PATH, game)
        os.makedirs(game_pred_dir, exist_ok=True)
        pred_file = os.path.join(game_pred_dir, "results_spotting.json")
        
        output_data = {
            "UrlLocal": game,
            "predictions": pred_list
        }
        with open(pred_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved prediction file: {pred_file}")

    print("\n" + "="*50)
    print("RUNNING SOCCERNET ACTION SPOTTING EVALUATION")
    print("="*50)

    # 1. Custom evaluation for first half only (True Performance on local videos)
    targets_half1 = []
    detections_half1 = []
    closests_half1 = []

    for game in local_valid_games:
        # Load ground truth
        label_file = os.path.join(ROOT_PATH, game, "Labels-v2.json")
        labels = json.load(open(label_file))
        label_h1, _ = label2vector(labels, num_classes=17, version=2, EVENT_DICTIONARY=EVENT_DICTIONARY_V2, framerate=2)
        
        # Load predictions
        pred_file = os.path.join(PREDS_PATH, game, "results_spotting.json")
        predictions = json.load(open(pred_file))
        pred_h1, _ = predictions2vector(predictions, num_classes=17, version=2, EVENT_DICTIONARY=EVENT_DICTIONARY_V2, framerate=2)

        targets_half1.append(label_h1)
        detections_half1.append(pred_h1)

        closest = np.zeros(label_h1.shape) - 1
        for c in range(label_h1.shape[-1]):
            indexes = np.where(label_h1[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest.shape[0])
            for i in range(len(indexes) - 2):
                idx = i + 1
                start = max(0, (indexes[idx - 1] + indexes[idx]) // 2)
                stop = min(closest.shape[0], (indexes[idx] + indexes[idx + 1]) // 2)
                closest[start:stop, c] = label_h1[indexes[idx], c]
        closests_half1.append(closest)

    # Compute custom first half mAP (1s, 2s, 3s, 4s, 5s tolerances)
    deltas = np.array([1, 2, 3, 4, 5])
    a_mAP_h1, a_mAP_per_class_h1, _, _, _, _ = average_mAP(targets_half1, detections_half1, closests_half1, framerate=2, deltas=deltas)

    print("\n>>> EVALUATION RESULT (First Half Only - 5 Local Videos) <<<")
    print(f"Average mAP (1s-5s tolerances): {a_mAP_h1 * 100:.2f}%")
    print("\nPer-class mAP:")
    class_names = sorted(EVENT_DICTIONARY_V2.keys(), key=lambda x: EVENT_DICTIONARY_V2[x])
    for idx, c_name in enumerate(class_names):
        print(f"  {c_name:<20}: {a_mAP_per_class_h1[idx] * 100:.2f}%")

    # 2. Official Evaluation (Both halves - SoccerNet standard evaluation)
    print("\n>>> EVALUATION RESULT (Full Match - 5 Local Videos, Second Half empty) <<<")
    original_getListGames = AS.getListGames
    AS.getListGames = lambda split, **kwargs: local_valid_games
    try:
        results = AS.evaluate(
            SoccerNet_path=ROOT_PATH,
            Predictions_path=PREDS_PATH,
            prediction_file="results_spotting.json",
            split="valid",
            version=2,
            framerate=2,
            metric="tight",  # tight evaluates at [1, 2, 3, 4, 5] seconds
            EVENT_DICTIONARY=EVENT_DICTIONARY_V2
        )
        print(f"Official format Average mAP (1s-5s tolerances): {results['a_mAP'] * 100:.2f}%")
    except Exception as e:
        print(f"Failed to run official evaluate: {e}")
    finally:
        AS.getListGames = original_getListGames

if __name__ == "__main__":
    main()
