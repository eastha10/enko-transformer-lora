from src.model import make_model
from src.data import build_dataloader
from src.data import make_batch

import os
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

        if (step + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch}] "
                f"step: {step + 1}, "
                f"loss: {loss.item():.4f}, "
                f"avg_loss: {avg_loss:.4f}"
            )

    return total_loss / total_steps

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

def main():
    save_dir = "checkpoints"

    ## 수정: 최종적으로 몇 epoch까지 학습할지 정해야 함
    ## 예: checkpoint-epoch-3.pt가 있으면 4부터 시작해서 5까지 학습
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

    model = make_model(
        src_vocab=16000,
        tgt_vocab=16000,
        d_model=256,
        N_en=6,
        N_de=3,
        d_ff=1024,
        h=4,
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=5e-5, eps=1e-9)

    ## 수정: optimizer까지 만든 뒤에 checkpoint를 불러와야 함
    ## 이유: checkpoint 안에 optimizer_state_dict도 들어있기 때문
    start_epoch = load_checkpoint_if_exists(
        model=model,
        optimizer=optimizer,
        save_dir=save_dir,
        device=device
    )

    ## 수정: 이미 목표 epoch까지 학습된 경우 바로 종료
    ## 예: checkpoint-epoch-5.pt가 있는데 total_epochs=5면 더 학습할 필요 없음
    if start_epoch > total_epochs:
        print(f"Already trained up to epoch {start_epoch - 1}.")
        return

    ## 수정: range(1, epochs + 1)이 아니라 start_epoch부터 시작해야 함
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

        ## 수정: valid_loader를 만들었으니 epoch마다 evaluate 실행
        valid_loss = evaluate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device
        )

        print(f"[Epoch {epoch}] valid_loss: {valid_loss:.4f}")

        ## 수정: checkpoint 저장은 train_loss, valid_loss가 계산된 뒤에 해야 함
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