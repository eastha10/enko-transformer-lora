from src.model import make_model
from src.data import build_dataloader
from src.data import make_batch
from src.checkpoint import save_checkpoint, load_checkpoint_if_exists
from src.model.lora import load_baseline_weights_into_lora_model

import torch
import argparse
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

import time
import csv
import os

def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def format_seconds(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.2f}s"

def compute_loss(model, batch, criterion, device):
    src_ids, tgt_ids = batch

    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)

    src, tgt_input, tgt_y, src_mask, tgt_mask = make_batch(src_ids, tgt_ids)

    out = model(src, tgt_input, src_mask, tgt_mask)
    log_probs = model.generator(out)

    loss = criterion(
        log_probs.reshape(-1, log_probs.size(-1)),
        tgt_y.reshape(-1)
    )

    return loss

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    grad_clip=1.0,
    max_steps=None
):
    model.train()

    total_loss = 0.0
    total_steps = 0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync_cuda(device)
    start_time = time.perf_counter()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        total=len(train_loader),
        dynamic_ncols=True
    )

    for step, batch in enumerate(progress_bar, start=1):
        if max_steps is not None and step > max_steps:
            break

        src_ids, tgt_ids = batch
        batch_size = src_ids.size(0)

        optimizer.zero_grad()

        loss = compute_loss(model, batch, criterion, device)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip
            )

        optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        total_samples += batch_size

        avg_loss = total_loss / total_steps

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

    sync_cuda(device)
    end_time = time.perf_counter()

    train_time = end_time - start_time
    avg_loss = total_loss / total_steps
    avg_step_time = train_time / total_steps
    samples_per_sec = total_samples / train_time

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = None

    return {
        "train_loss": avg_loss,
        "train_time": train_time,
        "avg_step_time": avg_step_time,
        "samples_per_sec": samples_per_sec,
        "total_steps": total_steps,
        "total_samples": total_samples,
        "peak_memory_mb": peak_memory_mb
    }

@torch.no_grad()
def evaluate(model, valid_loader, criterion, device, max_steps=None):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(tqdm(valid_loader, desc="Evaluate")):
        if max_steps is not None and step >= max_steps:
            break

        loss = compute_loss(model, batch, criterion, device)

        total_loss += loss.item()
        total_steps += 1

    return total_loss / total_steps

def freeze_all_parameters(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_lora_parameters(model):
    for name, p in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.requires_grad = True

def print_trainable_parameters(model, verbose=False):
    trainable = 0
    total = 0

    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if verbose:
                print(name, p.numel())

    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")

def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required checkpoint not found: {path}")

def build_model(mode, device):
    if mode in ["baseline", "fft"]:
        model = make_model(
            src_vocab=16000,
            tgt_vocab=16000,
            d_model=256,
            N_en=6,
            N_de=3,
            d_ff=1024,
            h=4,
            dropout=0.1,
            use_lora=False
        )

    elif mode == "lora":
        model = make_model(
            src_vocab=16000,
            tgt_vocab=16000,
            d_model=256,
            N_en=6,
            N_de=3,
            d_ff=1024,
            h=4,
            dropout=0.1,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            lora_targets=("q", "v")
        )

        freeze_all_parameters(model)
        unfreeze_lora_parameters(model)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model.to(device)

def build_optimizer(model, mode):
    if mode in ["baseline", "fft"]:
        return Adam(
            model.parameters(),
            lr=5e-5,
            eps=1e-9
        )

    elif mode == "lora":
        return Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5,
            eps=1e-9
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def get_save_dir(mode):
    if mode == "baseline":
        return "checkpoints"

    elif mode == "fft":
        return "checkpoints/fft"

    elif mode == "lora":
        return "checkpoints/lora"

    else:
        raise ValueError(f"Unknown mode: {mode}")

def save_epoch_metrics(metrics_path, row):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    file_exists = os.path.exists(metrics_path)

    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def load_baseline_weights(model, baseline_path, device):
    checkpoint = torch.load(baseline_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Baseline weights loaded from: {baseline_path}")
    print(f"Baseline epoch: {checkpoint.get('epoch', 'unknown')}")

    return model
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "fft", "lora"]
    )
    args = parser.parse_args()

    mode = args.mode
    save_dir = get_save_dir(mode)

    print(f"Mode: {mode}")
    print(f"Save dir: {save_dir}")

    total_epochs = 5

    train_loader = build_dataloader(
        parquet_path='data/train.parquet',
        src_spm_path='tokenizer/src_spm.model',
        tgt_spm_path='tokenizer/tgt_spm.model',
        batch_size=32,
        shuffle=True
    )

    valid_loader = build_dataloader(
        parquet_path='data/valid.parquet',
        src_spm_path='tokenizer/src_spm.model',
        tgt_spm_path='tokenizer/tgt_spm.model',
        batch_size=32,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(mode, device)
    print_trainable_parameters(model, verbose=(mode == "lora"))

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = build_optimizer(model, mode)

    if mode == "baseline":
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            save_dir=save_dir,
            device=device
        )

    elif mode == "fft":
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            save_dir=save_dir,
            device=device
        )

        if start_epoch == 1:
            baseline_path = "checkpoints/checkpoint-epoch-5.pt"
            check_file_exists(baseline_path)
            model = load_baseline_weights(
                model=model,
                baseline_path=baseline_path,
                device=device
            )

    elif mode == "lora":
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            save_dir=save_dir,
            device=device
        )

        if start_epoch == 1:
            baseline_path = "checkpoints/checkpoint-epoch-5.pt"
            check_file_exists(baseline_path)
            baseline_epoch = load_baseline_weights_into_lora_model(
                lora_model=model,
                baseline_path=baseline_path,
                device=device
            )
            print(f"Initialized LoRA model from baseline epoch {baseline_epoch}")

    if start_epoch > total_epochs:
        print(f"Already trained up to epoch {start_epoch - 1}.")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        train_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        train_loss = train_result["train_loss"]

        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}")
        print(f"[Epoch {epoch}] train_time: {format_seconds(train_result['train_time'])}")
        print(f"[Epoch {epoch}] avg_step_time: {train_result['avg_step_time']:.4f}s")
        print(f"[Epoch {epoch}] samples/sec: {train_result['samples_per_sec']:.2f}")

        if train_result["peak_memory_mb"] is not None:
            print(f"[Epoch {epoch}] peak_memory: {train_result['peak_memory_mb']:.2f} MB")

        valid_loss = evaluate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device
        )

        print(f"[Epoch {epoch}] valid_loss: {valid_loss:.4f}")

        save_epoch_metrics(
            metrics_path=f"logs/{mode}_metrics.csv",
            row={
                "mode": mode,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_time_sec": train_result["train_time"],
                "avg_step_time_sec": train_result["avg_step_time"],
                "samples_per_sec": train_result["samples_per_sec"],
                "total_steps": train_result["total_steps"],
                "total_samples": train_result["total_samples"],
                "peak_memory_mb": train_result["peak_memory_mb"]
            }
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            save_dir=save_dir
        )

if __name__ == "__main__":
    main()