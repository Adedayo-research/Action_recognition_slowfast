import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is on: {device}")

# 1. ARCHITECTURE SETUP & WEIGHT CONFIGURATIONS

yolo_weights = r'C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_1\assets\yolov8n.pt'
yolo_model = YOLO(yolo_weights)

# Auto-detect the latest specialized SlowFast weights
ckpt_dir = r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_2\checkpoints\exp2d_unfrozen_blocks4_5"
list_of_weights = glob.glob(os.path.join(ckpt_dir, "*.pth"))
if not list_of_weights:
    print(f"ERROR: No trained .pth weights found in {ckpt_dir}! Cannot proceed.")
    exit()
slowfast_weights = max(list_of_weights, key=os.path.getctime)
print(f"Found latest specialized weights: {slowfast_weights}")

action_labels = {
    0: "Background", 1: "Ball out of play", 2: "Clearance", 3: "Corner",
    4: "Direct free-kick", 5: "Foul", 6: "Goal", 7: "Indirect free-kick",
    8: "Injury", 9: "Kick-off", 10: "Offside", 11: "Penalty",
    12: "Red card", 13: "Shots off target", 14: "Shots on target",
    15: "Substitution", 16: "Throw-in", 17: "Yellow card"
}
num_classes = 18

print("Initializing baseline SlowFast model layout from Torch Hub...")
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)

# Freeze the backbone initially
for param in model.parameters():
    param.requires_grad = False

# Match the architecture by unfreezing blocks 4 and 5
for param in model.blocks[4].parameters():
    param.requires_grad = True
for param in model.blocks[5].parameters():
    param.requires_grad = True

# Swap out head projection
in_dim = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features=in_dim, out_features=num_classes)

# Load your specialized trained model weights
if os.path.exists(slowfast_weights):
    model.load_state_dict(torch.load(slowfast_weights, map_location=device))
    print(f"Successfully loaded specialized SlowFast weights from: {slowfast_weights}")
else:
    print(f"ERROR: '{slowfast_weights}' not found. Cannot proceed.")
    exit()

model = model.to(device).eval()

# Match training preprocessing exactly (Scale to 0-1, but NO ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 2. FILE STREAM CHANNELS DEFINITION

input_video = r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_1\public_dataset\val\Red card\Red Card6.avi"
output_video = "crystal_palace_live_tracking_new1.mp4"

cap = cv2.VideoCapture(input_video)

success, test_frame = cap.read()
if not success:
    print("ERROR: Cannot read frames from the input video file.")
    exit()

height, width = test_frame.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Sliding Windows rolling queues
global_frame_buffer = []
global_current_action = "Analyzing..."

print(f"\nProcessing Broadcast Feed: {input_video} ({width}x{height} at {fps} FPS)...")
frames_processed = 0


# 3. FRAME PROCESSING LOOP

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frames_processed += 1
        if frames_processed % 30 == 0:
            print(f"  ... Processed {frames_processed} frames ...")

        # 1. Standardize and preprocess the FULL broadcast frame view
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor_img = transform(pil_img)
        global_frame_buffer.append(tensor_img)

        # 2. Process temporal context via SlowFast every 32 frames
        if len(global_frame_buffer) == 32:
            try:
                video_tensor = torch.stack(global_frame_buffer, dim=1)
                fast_tensor = video_tensor.unsqueeze(0).to(device)
                slow_tensor = video_tensor[:, ::4, :, :].unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model([slow_tensor, fast_tensor])
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                    action_name = action_labels.get(predicted_idx.item(), "Background")
                    conf_score = confidence.item() * 100

                    if action_name != "Background" and conf_score > 40.0:
                        global_current_action = f"{action_name} ({conf_score:.0f}%)"
                    else:
                        global_current_action = "Normal Play"
            except Exception as e:
                print(f"Prediction Error: {e}")

            # STROBE OPTIMIZATION: Instead of popping 1 frame (which causes SlowFast to run every single frame),
            # pop 8 frames. This means the heavy 3D CNN will only execute ~4 times per second.
            del global_frame_buffer[:8]

        # 3. Spatial object tracking loops
        results = yolo_model.track(frame, persist=True, verbose=False, classes=[0])

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # Drawing thin tracking squares around players
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)


        # 4. DRAW HUD AND WRITE FRAME EXACTLY ONCE PER LOOP

        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (width, 50), (0, 0, 0), -1)

        # Apply clean 60% overlay transparency mix
        frame = cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0)

        # Render anti-aliased studio caption
        hud_text = f"Live Action Spotted: {global_current_action}"
        cv2.putText(frame, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Save single output frame safely to your output file
        out.write(frame)

except Exception as e:
    print(f"\nProcessing error: {e}")

finally:
    cap.release()
    out.release()
    print(f"\nPIPELINE COMPLETE: Processed {frames_processed} frames.")
    print(f"Video file securely compiled as '{output_video}'")
