# ============================================================
# reorganise.ps1
# Neatly restructures the Action_recognition_slowfast research
# folder for PhD archiving and reproducibility.
# Run from: Action_recognition_slowfast\
# ============================================================

$root = $PSScriptRoot

# ── 1. TOP-LEVEL: move stray root files into experiment_1 ───────────────────

$rootStray = @(
    "best_football_action_model.pth",
    "football_slowfast.ipynb"
)

foreach ($f in $rootStray) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        Move-Item $src (Join-Path $root "experiment_1\$f") -Force
        Write-Host "[MOVED]  $f  →  experiment_1\"
    }
}

# ── 2. EXPERIMENT_1: tidy up ─────────────────────────────────────────────────

$exp1 = Join-Path $root "experiment_1"
$null = New-Item -ItemType Directory -Force (Join-Path $exp1 "outputs")
$null = New-Item -ItemType Directory -Force (Join-Path $exp1 "scripts")

# Move output videos
foreach ($f in @("output.avi","output2.avi","output2.mp4")) {
    $src = Join-Path $exp1 $f
    if (Test-Path $src) {
        Move-Item $src (Join-Path $exp1 "outputs\$f") -Force
        Write-Host "[MOVED]  experiment_1\$f  →  experiment_1\outputs\"
    }
}

# Move python scripts
foreach ($f in @("football_action_detection.py","football_action_recognition.py")) {
    $src = Join-Path $exp1 $f
    if (Test-Path $src) {
        Move-Item $src (Join-Path $exp1 "scripts\$f") -Force
        Write-Host "[MOVED]  experiment_1\$f  →  experiment_1\scripts\"
    }
}

# ── 3. EXPERIMENT_2: create organised sub-directories ────────────────────────

$exp2 = Join-Path $root "experiment_2"

$dirs = @(
    "scripts",
    "checkpoints\exp2a_frozen_crossentropy",
    "checkpoints\exp2b_frozen_focal",
    "checkpoints\exp2c_unfrozen",
    "checkpoints\archived_early_sweeps",
    "predictions",
    "evaluation",
    "assets"
)

foreach ($d in $dirs) {
    $null = New-Item -ItemType Directory -Force (Join-Path $exp2 $d)
}

# ── 4. Move Python scripts ───────────────────────────────────────────────────

$scripts = @(
    "training_frozen_backbone.py",
    "training_unfrozen_backbone.py",
    "detector.py",
    "evaluate_mAP.py",
    "investigate_dataset.py",
    "load_soccerNet_action.py"
)

foreach ($f in $scripts) {
    $src = Join-Path $exp2 $f
    if (Test-Path $src) {
        Move-Item $src (Join-Path $exp2 "scripts\$f") -Force
        Write-Host "[MOVED]  $f  →  scripts\"
    }
}

# Move notebook (keep for reference)
$nb = Join-Path $exp2 "training_unfrozen_backbone.ipynb"
if (Test-Path $nb) {
    Move-Item $nb (Join-Path $exp2 "scripts\training_unfrozen_backbone.ipynb") -Force
    Write-Host "[MOVED]  training_unfrozen_backbone.ipynb  →  scripts\"
}

# ── 5. Move Checkpoints (Exp 2a: Cross-Entropy) ──────────────────────────────

Get-ChildItem $exp2 -Filter "fixed_bs16_lr0.0001_cross_entropy_epoch_*.pth" | ForEach-Object {
    Move-Item $_.FullName (Join-Path $exp2 "checkpoints\exp2a_frozen_crossentropy\$($_.Name)") -Force
    Write-Host "[MOVED]  $($_.Name)  →  checkpoints\exp2a_frozen_crossentropy\"
}

# ── 6. Move Checkpoints (Exp 2b: Focal Loss) ─────────────────────────────────

Get-ChildItem $exp2 -Filter "fixed_bs16_lr0.0001_focal_loss_epoch_*.pth" | ForEach-Object {
    Move-Item $_.FullName (Join-Path $exp2 "checkpoints\exp2b_frozen_focal\$($_.Name)") -Force
    Write-Host "[MOVED]  $($_.Name)  →  checkpoints\exp2b_frozen_focal\"
}

# ── 7. Move early/archived sweep checkpoints ─────────────────────────────────

$archived = @(
    "epoch_tqdm_bs16_lr0.0001_epoch_*.pth",
    "grateful-sweep-1_epoch_*.pth",
    "silvery-sweep-1_epoch_*.pth",
    "warm-sweep-1_epoch_*.pth",
    "run_bs16_lr0.0001_epoch_wise_epoch_*.pth",
    "model_epoch_*.pth"
)

foreach ($pattern in $archived) {
    Get-ChildItem $exp2 -Filter $pattern | ForEach-Object {
        Move-Item $_.FullName (Join-Path $exp2 "checkpoints\archived_early_sweeps\$($_.Name)") -Force
        Write-Host "[MOVED]  $($_.Name)  →  checkpoints\archived_early_sweeps\"
    }
}

# ── 8. Move YOLO / asset files ───────────────────────────────────────────────

foreach ($f in @("yolo_8_best.pt","yolov8n.pt","crystal_palace_live_tracking_new.mp4")) {
    $src = Join-Path $exp2 $f
    if (Test-Path $src) {
        Move-Item $src (Join-Path $exp2 "assets\$f") -Force
        Write-Host "[MOVED]  $f  →  assets\"
    }
}

# ── 9. Move evaluation outputs ───────────────────────────────────────────────

$evalSrc = Join-Path $exp2 "predictions"
if (Test-Path $evalSrc) {
    # predictions folder already exists, just move its contents into evaluation\
    Get-ChildItem $evalSrc -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($evalSrc.Length + 1)
        $dest = Join-Path $exp2 "evaluation\predictions\$rel"
        $null = New-Item -ItemType Directory -Force (Split-Path $dest)
        Move-Item $_.FullName $dest -Force
    }
    Remove-Item $evalSrc -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[MOVED]  predictions\  →  evaluation\predictions\"
}

# ── 10. Move experiment_1 YOLO model ─────────────────────────────────────────

$yolo1 = Join-Path $exp1 "yolov8n.pt"
if (Test-Path $yolo1) {
    $null = New-Item -ItemType Directory -Force (Join-Path $exp1 "assets")
    Move-Item $yolo1 (Join-Path $exp1 "assets\yolov8n.pt") -Force
    Write-Host "[MOVED]  experiment_1\yolov8n.pt  →  experiment_1\assets\"
}

# ── 11. Write a README for the project ──────────────────────────────────────

$readme = @"
# Action Recognition – SlowFast (SoccerNet)
PhD Research Experiments — Football Action Spotting

## Project Structure

```
Action_recognition_slowfast/
│
├── experiment_1/                  # Prototype: YOLO + single-video detection
│   ├── scripts/                   # football_action_detection.py, football_action_recognition.py
│   ├── assets/                    # YOLO weights
│   ├── outputs/                   # Output videos from experiment 1
│   └── public_dataset/            # Public dataset used for exp 1
│
├── experiment_2/                  # Main: SlowFast + SoccerNet benchmark
│   ├── scripts/                   # All Python training & evaluation scripts
│   │   ├── training_frozen_backbone.py      # Exp 2a & 2b (Frozen backbone sweeps)
│   │   ├── training_unfrozen_backbone.py    # Exp 2c (Unfrozen backbone fine-tuning)
│   │   ├── detector.py                      # Sliding-window inference / detection
│   │   ├── evaluate_mAP.py                  # Official SoccerNet mAP evaluation
│   │   ├── investigate_dataset.py           # Label distribution analysis
│   │   └── load_soccerNet_action.py         # Dataset loader helper
│   │
│   ├── checkpoints/
│   │   ├── exp2a_frozen_crossentropy/       # Frozen backbone, Cross-Entropy loss
│   │   ├── exp2b_frozen_focal/              # Frozen backbone, Focal loss (best baseline)
│   │   ├── exp2c_unfrozen/                  # Unfrozen backbone checkpoints (in progress)
│   │   └── archived_early_sweeps/           # Discarded early sweep checkpoints
│   │
│   ├── evaluation/
│   │   └── predictions/                     # JSON prediction files for mAP scoring
│   │
│   ├── assets/                              # YOLO weights, demo videos
│   ├── SoccerNet_Dataset/                   # Raw SoccerNet local data partition
│   └── wandb/                               # W&B sweep logs
```

## Experiment Log

| ID   | Name                         | Loss          | Val Acc | Avg mAP | Notes                          |
|------|------------------------------|---------------|---------|---------|-------------------------------|
| 2a   | Frozen Backbone, CE Loss     | Cross-Entropy | 21.32%  | –       | Baseline; majority-class bias  |
| 2b   | Frozen Backbone, Focal Loss  | Focal (γ=2)   | 32.99%  | 1.34%   | +11% over CE; backbone ceiling |
| 2c   | Unfrozen Backbone (blocks[5])| Focal (γ=2)   | TBD     | TBD     | Currently training             |

## Key Dependencies
- PyTorch + CUDA
- SlowFast R50 (torch.hub)
- SoccerNet Python SDK
- Weights & Biases (wandb)
"@

Set-Content -Path (Join-Path $root "README.md") -Value $readme -Encoding UTF8
Write-Host "[CREATED] README.md at project root"

Write-Host ""
Write-Host "=============================================="
Write-Host " Reorganisation complete!"
Write-Host "=============================================="
