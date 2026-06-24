import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# PHASE 1:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Computing on: {device}\n")


class PublicVideoDataset(Dataset):
    def __init__(self, root_dir, alpha=4, fast_frames=32):
        self.root_dir = root_dir
        self.alpha = alpha
        self.fast_frames = fast_frames

        # DATA PREPROCESSING PIPELINE
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.video_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.endswith(('.mp4', '.avi')):
                        self.video_paths.append(os.path.join(cls_dir, file))
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.video_paths)


    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        processed_frames = []

        for _ in range(self.fast_frames):
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                tensor_img = self.transform(pil_img)
                processed_frames.append(tensor_img)
            else:
                processed_frames.append(torch.zeros((3, 224, 224)))
        cap.release()

        video_tensor = torch.stack(processed_frames, dim=1)
        fast_tensor = video_tensor
        slow_tensor = video_tensor[:, ::self.alpha, :, :]

        return [slow_tensor, fast_tensor], label, video_path


train_dir = r"/Action_recognition_slowfast/public_dataset/train"
val_dir = r"/Action_recognition_slowfast/public_dataset/val"

train_dataset = PublicVideoDataset(root_dir=train_dir)
val_dataset = PublicVideoDataset(root_dir=val_dir)
action_classes = train_dataset.classes
num_classes = len(action_classes)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)


# PHASE 2: MODEL ARCHITECTURE

model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

in_features = model.blocks[6].proj.in_features
model.blocks[6].proj = nn.Linear(in_features=in_features, out_features=num_classes)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# PHASE 3: THE TRAINING PHASE
epochs = 20

history_train_loss = []
history_val_accuracy = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs} ---")


    # Training

    model.train()
    running_loss = 0.0

    for videos, labels, _ in train_loader:
        slow = videos[0].to(device)
        fast = videos[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model([slow, fast])
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Logging the loss
    epoch_loss = running_loss / len(train_loader)
    print(f"  Training Loss: {epoch_loss:.4f}")
    history_train_loss.append(epoch_loss)


    # Validation Phase

    model.eval()
    correct_guesses = 0
    total_videos = 0

    with torch.no_grad():
        for videos, labels, _ in val_loader:
            slow = videos[0].to(device)
            fast = videos[1].to(device)
            labels = labels.to(device)

            predictions = model([slow, fast])
            best_guess = torch.argmax(predictions, dim=1)

            correct_guesses += (best_guess == labels).sum().item()
            total_videos += labels.size(0)

    # Log the accuracy for this epoch
    accuracy = (correct_guesses / total_videos) * 100
    print(f"  Validation Accuracy: {accuracy:.2f}%")
    history_val_accuracy.append(accuracy)

print("Training Complete")
torch.save(model.state_dict(), "../best_football_action_model.pth")
print("Weights saved")


# VISUALIZATION

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
epochs_range = range(1, epochs + 1)

# Plot 1: Training Loss
ax1.plot(epochs_range, history_train_loss, label='Training Loss', color='red', marker='o', linewidth=2)
ax1.set_title('Model Error During Training')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross Entropy Loss')
ax1.set_xticks(epochs_range)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Plot 2: Validation Accuracy
ax2.plot(epochs_range, history_val_accuracy, label='Validation Accuracy', color='blue', marker='s', linewidth=2)
ax2.set_title('Model Accuracy on Unseen Clips')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_xticks(epochs_range)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.show()


# TESTING UNSEEN VIDEO

test_video_path = val_dataset.video_paths[0]
print(f"File: {test_video_path}")

model.eval()

cap = cv2.VideoCapture(test_video_path)
frames = []

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for _ in range(32):
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor_img = test_transform(pil_img)
        frames.append(tensor_img)
    else:
        frames.append(torch.zeros((3, 224, 224)))
cap.release()

video_tensor = torch.stack(frames, dim=1).unsqueeze(0)

fast_tensor = video_tensor.to(device)
slow_tensor = video_tensor[:, :, ::4, :, :].to(device)

with torch.no_grad():
    outputs = model([slow_tensor, fast_tensor])
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

predicted_action = action_classes[predicted_idx.item()]

print(f"Player Action Prediction: {predicted_action.upper()}")
print(f"Confidence:     {confidence.item() * 100:.2f}%")
