
import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from SoccerNet.utils import getListGames
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True

# Raw path targeting your verified local database partition folder location
ROOT_PATH = r"C:\Users\omoni\Desktop\experiments\Action_recognition_slowfast\experiment_2\SoccerNet_Dataset"



# DATASET PROCESSING

class SoccerNetActionDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.games = getListGames(split=split)
        self.data_map = []

        # Official 18-Class Mapping Schema (Index 0 is Background)
        self.label_to_id = {
            "Background": 0, "Ball out of play": 1, "Clearance": 2, "Corner": 3,
            "Direct free-kick": 4, "Foul": 5, "Goal": 6, "Indirect free-kick": 7,
            "Injury": 8, "Kick-off": 9, "Offside": 10, "Penalty": 11,
            "Red card": 12, "Shots off target": 13, "Shots on target": 14,
            "Substitution": 15, "Throw-in": 16, "Yellow card": 17
        }
        self._build_index()

    def _build_index(self):
        print(f"Indexing {self.split} set...")
        count = 0
        for game in self.games:
            game_path = os.path.join(self.root_dir, game)
            json_path = os.path.join(game_path, "Labels-v2.json")
            video_path = os.path.join(game_path, "1_224p.mkv")

            if os.path.exists(json_path) and os.path.exists(video_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for annotation in data['annotations']:
                        if annotation['label'] in self.label_to_id:
                            self.data_map.append({
                                'label': annotation['label'],
                                'gameTime': annotation['gameTime'],
                                'video_path': video_path,
                            })
                            count += 1
        print(f"Safely indexed {count} action points in {self.split} split.")

    def _load_video_clip(self, video_path, frame_idx, num_frames=32):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Colourspace fix
            frames.append(frame)
        cap.release()
        clip = np.array(frames).astype(np.float32) / 255.0
        clip = clip.transpose(3, 0, 1, 2)  # [C, T, H, W]
        return torch.from_numpy(clip)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        sample = self.data_map[idx]
        try:
            # SoccerNet gameTime format: "1 - 00:34" (half - MM:SS)
            parts = sample['gameTime'].split(' - ')
            timestamp = parts[1] if len(parts) > 1 else parts[0]
            minutes, seconds = map(float, timestamp.split(':'))
            total_seconds = minutes * 60 + seconds
        except (IndexError, ValueError):
            total_seconds = 0.0

        fps = 25.0
        start_frame = max(0, int((total_seconds - 1.0) * fps))
        video_tensor = self._load_video_clip(sample['video_path'], start_frame, num_frames=32)

        fast_path = video_tensor
        slow_path = video_tensor[:, ::4, :, :]
        label_id = self.label_to_id.get(sample['label'], 0)
        return [slow_path, fast_path], torch.tensor(label_id, dtype=torch.long)

    def get_label_distribution(self):
        """Returns a Counter of label IDs for computing class weights."""
        counts = Counter()
        for sample in self.data_map:
            label_id = self.label_to_id.get(sample['label'], 0)
            counts[label_id] += 1
        return counts



# MODEL

def get_model(num_classes):
    print("Loading pretrained SlowFast R50 from Torch Hub...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    # Lock down weights in the 3D CNN body to prevent memory overflow
    for param in model.parameters():
        param.requires_grad = False

    # Swap out head projection layer
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features=in_features, out_features=num_classes)
    return model



# 3. FOCAL LOSS FUNCTION

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



#SWEEP CONFIGURATION

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_accuracy',  # Matches your goal perfectly
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {'value': 16},
        'learning_rate': {'value': 0.0001},

        # FIXED: Increased to let your classification head learn rare patterns
        'epochs': {'value': 15},

        "architecture": {'value': "SlowFast"},
        'loss_type': {'values': ["cross_entropy", "focal_loss"]},
        'action_weight_multiplier': {'value': 4.0}, #(, 10.0, 15.0]
        'focal_gamma': {'values': [1.5]}, #, 2.0, 3.0]},
        "dataset": {'value': "SoccerNet-v2"}
    }
}

sweep_id = wandb.sweep(sweep_config, project="slowfast_frozen_backbone_fixed")



# VALIDATION

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            slow_tensor = inputs[0].to(device)
            fast_tensor = inputs[1].to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model([slow_tensor, fast_tensor])
                loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Compute raw evaluation accuracy values
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predictions == targets).item()
            total_samples += targets.size(0)

    avg_loss = val_loss / len(val_loader)
    accuracy_score = correct_predictions / total_samples
    return avg_loss, accuracy_score



# TRAINING

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, criterion, device):
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)):
            optimizer.zero_grad()

            slow_tensor = inputs[0].to(device, non_blocking=True)
            fast_tensor = inputs[1].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model([slow_tensor, fast_tensor])
                loss = criterion(outputs, targets)


            # Execute backward pass math and optimize weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Validation checks
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)


        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        print(f"--- Epoch {epoch} Complete | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc * 100:.2f}% ---")

        # Save model checkpoint weight
        run_name = wandb.run.name if wandb.run else "local"
        torch.save(model.state_dict(), f"{run_name}_epoch_{epoch}.pth")



# MAIN SWEEP WRAPPER

def compute_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights so rare classes get higher loss."""
    dist = dataset.get_label_distribution()
    total = sum(dist.values())
    weights = torch.ones(num_classes)
    for class_id, count in dist.items():
        weights[class_id] = total / (num_classes * count) if count > 0 else 1.0
    print(f"\nComputed class weights (inverse-frequency):")
    for cid in range(num_classes):
        print(f"  Class {cid}: weight={weights[cid]:.3f} (count={dist.get(cid, 0)})")
    return weights.to(device)


def train():
    with wandb.init() as run:
        config = wandb.config
        wandb.run.name = f"fixed_bs{config.batch_size}_lr{config.learning_rate}_{config.loss_type}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n--- Initializing Active Sweep Run on Target Device: {device} ---")

        # Load 18-class output layout structure
        model = get_model(num_classes=18).to(device)

        # Build local structured dataset splits
        train_ds = SoccerNetActionDataset(root_dir=ROOT_PATH, split="train")
        val_ds = SoccerNetActionDataset(root_dir=ROOT_PATH, split="valid")


        # MAIN SWEEP WRAPPER DATA PIPELINE

        # Setup DataLoaders (num_workers=2 to avoid CUDA pinned memory crash)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=False)

        # Compute proper inverse-frequency class weights from training data
        weight = compute_class_weights(train_ds, num_classes=18, device=device)

        # Only pass active, unfrozen layers to the optimizer
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_parameters, lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)

        # Select active loss function criterion mapping based on config
        if config.loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=weight)
        elif config.loss_type == "focal_loss":
            criterion = FocalLoss(gamma=config.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

        # Engage operations and run the training cycle
        train_model(model, train_loader, val_loader, optimizer, scheduler, config.epochs, criterion, device)


# SWEEP AGENT
if __name__ == '__main__':
    wandb.agent(sweep_id, function=train)
