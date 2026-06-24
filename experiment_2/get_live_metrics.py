import wandb

try:
    api = wandb.Api()
    run = api.run("25817442-edge-hill-university/slowfast_unfrozen_blocks4_5/p3xlmc5i")
    print(f"Epoch: {run.summary.get('epoch')}")
    print(f"Val Loss: {run.summary.get('val_loss'):.4f}")
    print(f"Val Accuracy: {run.summary.get('val_accuracy')*100:.2f}%")
    print(f"Val F1 Score: {run.summary.get('val_f1_score'):.4f}")
except Exception as e:
    print(f"Error fetching from wandb: {e}")
