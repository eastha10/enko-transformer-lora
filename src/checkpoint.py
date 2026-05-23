import os
import torch

def save_checkpoint(model, optimizer, epoch, train_loss, valid_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"checkpoint-epoch-{epoch}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        save_path
    )

    print(f"Checkpoint saved: {save_path}")

def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None

    checkpoint_files = [
        f for f in os.listdir(save_dir)
        if f.startswith("checkpoint-epoch-") and f.endswith(".pt")
    ]

    if len(checkpoint_files) == 0:
        return None

    checkpoint_files.sort(
        key=lambda x: int(x.replace("checkpoint-epoch-", "").replace(".pt", ""))
    )

    return os.path.join(save_dir, checkpoint_files[-1])

def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Checkpoint loaded for evaluation: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Train loss: {checkpoint.get('train_loss')}")
    print(f"Valid loss: {checkpoint.get('valid_loss')}")

    return model, checkpoint

def load_checkpoint_if_exists(model, optimizer, save_dir, device):
    checkpoint_path = find_latest_checkpoint(save_dir)

    if checkpoint_path is None:
        print("No checkpoint found. Start training from epoch 1.")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resume training from epoch {start_epoch}")

    return start_epoch