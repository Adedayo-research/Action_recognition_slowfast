# Action Recognition â€“ SlowFast (SoccerNet)
PhD Research Experiments â€” Football Action Spotting

## Project Structure

`
Action_recognition_slowfast/
â”‚
â”œâ”€â”€ experiment_1/                  # Prototype: YOLO + single-video detection
â”‚   â”œâ”€â”€ scripts/                   # football_action_detection.py, football_action_recognition.py
â”‚   â”œâ”€â”€ assets/                    # YOLO weights
â”‚   â”œâ”€â”€ outputs/                   # Output videos from experiment 1
â”‚   â””â”€â”€ public_dataset/            # Public dataset used for exp 1
â”‚
â”œâ”€â”€ experiment_2/                  # Main: SlowFast + SoccerNet benchmark
â”‚   â”œâ”€â”€ scripts/                   # All Python training & evaluation scripts
â”‚   â”‚   â”œâ”€â”€ training_frozen_backbone.py      # Exp 2a & 2b (Frozen backbone sweeps)
â”‚   â”‚   â”œâ”€â”€ training_unfrozen_backbone.py    # Exp 2c (Unfrozen backbone fine-tuning)
â”‚   â”‚   â”œâ”€â”€ detector.py                      # Sliding-window inference / detection
â”‚   â”‚   â”œâ”€â”€ evaluate_mAP.py                  # Official SoccerNet mAP evaluation
â”‚   â”‚   â”œâ”€â”€ investigate_dataset.py           # Label distribution analysis
â”‚   â”‚   â””â”€â”€ load_soccerNet_action.py         # Dataset loader helper
â”‚   â”‚
â”‚   â”œâ”€â”€ checkpoints/
â”‚   â”‚   â”œâ”€â”€ exp2a_frozen_crossentropy/       # Frozen backbone, Cross-Entropy loss
â”‚   â”‚   â”œâ”€â”€ exp2b_frozen_focal/              # Frozen backbone, Focal loss (best baseline)
â”‚   â”‚   â”œâ”€â”€ exp2c_unfrozen/                  # Unfrozen backbone checkpoints (in progress)
â”‚   â”‚   â””â”€â”€ archived_early_sweeps/           # Discarded early sweep checkpoints
â”‚   â”‚
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â””â”€â”€ predictions/                     # JSON prediction files for mAP scoring
â”‚   â”‚
â”‚   â”œâ”€â”€ assets/                              # YOLO weights, demo videos
â”‚   â”œâ”€â”€ SoccerNet_Dataset/                   # Raw SoccerNet local data partition
â”‚   â””â”€â”€ wandb/                               # W&B sweep logs
`

## Experiment Log

| ID   | Name                         | Loss          | Val Acc | Avg mAP | Notes                          |
|------|------------------------------|---------------|---------|---------|-------------------------------|
| 2a   | Frozen Backbone, CE Loss     | Cross-Entropy | 21.32%  | â€“       | Baseline; majority-class bias  |
| 2b   | Frozen Backbone, Focal Loss  | Focal (Î³=2)   | 32.99%  | 1.34%   | +11% over CE; backbone ceiling |
| 2c   | Unfrozen Backbone (blocks[5])| Focal (Î³=2)   | TBD     | TBD     | Currently training             |

## Key Dependencies
- PyTorch + CUDA
- SlowFast R50 (torch.hub)
- SoccerNet Python SDK
- Weights & Biases (wandb)
