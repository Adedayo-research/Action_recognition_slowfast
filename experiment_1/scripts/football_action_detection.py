import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is on: {device}")

# Load Pipeline
yolo_weights = 'yolov8n.pt'
yolo_model = YOLO(yolo_weights)
slowfast_weights = r'C:\Users\omoni\Desktop\experiments\best_football_model.pth'

action_labels = {0: "Red Card", 1: "Scoring", 2: "Tackling"}
num_classes = len(action_labels)

model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
model.blocks[6].proj = nn.Linear(in_features=model.blocks[6].proj.in_features, out_features=num_classes)

if os.path.exists(slowfast_weights):
    model.load_state_dict(torch.load(slowfast_weights, map_location=device))
    print("SlowFast weights loaded")
else:
    print(f"ERROR: '{slowfast_weights}' not found.")
    exit()

model = model.to(device).eval()

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



input_video = r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\public_dataset\val\Scoring\scoring 255.avi"
output_video = "crystal_palace_live_tracking.mp4"

cap = cv2.VideoCapture(input_video)

success, test_frame = cap.read()
if not success:
    print("ERROR: Cannot read frames from the video file")
    exit()

height, width = test_frame.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

# Reset the video back to frame 0 after our test
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- THE ROLLING BUFFERS ---
player_tensor_buffers = {}
player_current_actions = {}

print(f"\nProcessing Broadcast Feed: {input_video} ({width}x{height} at {fps} FPS)...")



# 3. FRAME-BY-FRAME STREAMING
frames_processed = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frames_processed += 1
        if frames_processed % 30 == 0:
            print(f"  ... Processed {frames_processed} frames ...")

        # Step A: Spatial Tracking (YOLO)
        results = yolo_model.track(frame, persist=True, verbose=False, classes=[0])

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):

                # Step B: The Dynamic Crop
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Skip invalid/empty boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_player = frame[y1:y2, x1:x2]

                if cropped_player.size == 0:
                    continue

                # Transform crop into a PyTorch Tensor
                pil_img = Image.fromarray(cv2.cvtColor(cropped_player, cv2.COLOR_BGR2RGB))
                tensor_img = transform(pil_img)

                # Initialize memory for new players
                if track_id not in player_tensor_buffers:
                    player_tensor_buffers[track_id] = []
                    player_current_actions[track_id] = "Analyzing..."  # Default UI text

                # Add new frame to player's memory queue
                player_tensor_buffers[track_id].append(tensor_img)

                # Step C: The 32-Frame Trigger (Temporal Classification)
                if len(player_tensor_buffers[track_id]) == 32:
                    try:
                        # Stacking memory into [Channels, Time, Height, Width]
                        video_tensor = torch.stack(player_tensor_buffers[track_id], dim=1)

                        # Split pathways
                        fast_tensor = video_tensor.unsqueeze(0).to(device)
                        slow_tensor = video_tensor[:, ::4, :, :].unsqueeze(0).to(device)

                        # Ask SlowFast for a prediction
                        with torch.no_grad():
                            outputs = model([slow_tensor, fast_tensor])
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)

                            action_name = action_labels[predicted_idx.item()]
                            conf_score = confidence.item() * 100

                            # Update the player's live status tag
                            player_current_actions[track_id] = f"{action_name} ({conf_score:.0f}%)"
                    except Exception as e:
                        print(f" Tensor error on player {track_id}: {e}")

                    # Critical Step: Remove the oldest frame to create the Sliding Window
                    player_tensor_buffers[track_id].pop(0)

                    # Step D: Draw CLEAN Graphical Overlays
                    # NO YOLO LABELS. Only draw your action labels.
                    current_action = player_current_actions.get(track_id, "Tracking...")

                    # Only draw if we have a confident action to show
                    if current_action != "Tracking...":
                        x1, y1, x2, y2 = map(int, box)

                        # Use a subtle box instead of thick yellow
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                        # Clean, professional HUD label
                        label = f"{current_action}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                        # Semi-transparent background for the label
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 5, y1), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Step E: Save compiled frame
        out.write(frame)

except Exception as e:
    print(f"\n Processing error: {e}")

finally:
    cap.release()
    out.release()
    if frames_processed > 0:
        print(f"PIPELINE COMPLETE: Processed {frames_processed} frames.")
        print(f"Video successfully saved to '{output_video}'")
    else:
        print("PIPELINE FAILED")