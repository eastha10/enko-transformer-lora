from src.model import make_model
from src.data import build_dataloader
from src.data import make_batch
from src.checkpoint import save_checkpoint, load_checkpoint_if_exists

import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

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
    log_interval=50,
    grad_clip=1.0,
    max_steps=None
):
    model.train()

    total_loss = 0.0
    total_steps = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        if max_steps is not None and step >= max_steps:
            break

        optimizer.zero_grad()

        loss = compute_loss(model, batch, criterion, device)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        avg_loss = total_loss / total_steps
        progress_bar.set_postfix({
            "loss": loss.item(),
            "avg_loss": avg_loss
        })

    return total_loss / total_steps

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

def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(name, p.numel())

    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")

def build_model(mode, device):
    if mode == "baseline":
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
    if mode == "baseline":
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

    elif mode == "lora":
        return "checkpoints/lora"

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def main():
    mode = "baseline"
    save_dir = get_save_dir(mode)
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
    print_trainable_parameters(model)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = build_optimizer(model, mode)

    start_epoch = load_checkpoint_if_exists(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir,
        device=device
    )

    if start_epoch > total_epochs:
        print(f"Already trained up to epoch {start_epoch - 1}.")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}")

        valid_loss = evaluate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device
        )

        print(f"[Epoch {epoch}] valid_loss: {valid_loss:.4f}")

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