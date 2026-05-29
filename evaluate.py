import math
import torch
import torch.nn as nn
import sentencepiece as spm
from tqdm import tqdm
import argparse

from src.data import build_dataloader, make_batch, make_src_mask
from src.model import make_model
from src.checkpoint import load_model_checkpoint
from src.model.transformer import subsequent_mask

from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score


PAD = 0
UNK = 1
BOS = 2
EOS = 3

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

TEST_PARQUET_PATH = "data/test.parquet"
SRC_SPM_PATH = "tokenizer/src_spm.model"
TGT_SPM_PATH = "tokenizer/tgt_spm.model"

MAX_LEN = 60
MAX_SAMPLES = None
SAMPLE_COUNT = 5

def get_save_dir(mode):
    if mode == "baseline":
        return "checkpoints"

    elif mode == "fft":
        return "checkpoints/fft"

    elif mode == "lora_r4":
        return "checkpoints/lora_r4"
    
    elif mode == "lora_r8":
        return "checkpoints/lora_r8"
    
    elif mode == "lora_r16":
        return "checkpoints/lora_r16"

    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_tokenizers(src_spm_path, tgt_spm_path):
    src_sp = spm.SentencePieceProcessor()
    tgt_sp = spm.SentencePieceProcessor()

    src_sp.load(src_spm_path)
    tgt_sp.load(tgt_spm_path)

    return src_sp, tgt_sp


def remove_special_tokens(token_ids):
    token_ids = list(token_ids)

    if len(token_ids) > 0 and token_ids[0] == BOS:
        token_ids = token_ids[1:]

    if EOS in token_ids:
        token_ids = token_ids[:token_ids.index(EOS)]

    token_ids = [
        token_id for token_id in token_ids
        if token_id not in [PAD, BOS, EOS]
    ]

    return token_ids


def evaluate_loss(model, dataloader, device):
    model.eval()

    criterion = nn.NLLLoss(ignore_index=PAD, reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            src, tgt_input, tgt_y, src_mask, tgt_mask = make_batch(src, tgt)

            out = model(src, tgt_input, src_mask, tgt_mask)
            log_probs = model.generator(out)

            loss = criterion(
                log_probs.reshape(-1, log_probs.size(-1)),
                tgt_y.reshape(-1)
            )

            non_pad_tokens = (tgt_y != PAD).sum().item()

            total_loss += loss.item()
            total_tokens += non_pad_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


def greedy_decoding(model, src, src_mask, max_len, start_symbol):
    model.eval()

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        generated_ids = torch.full(
            size=(1, 1),
            fill_value=start_symbol,
            dtype=torch.long,
            device=src.device
        )

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(generated_ids.size(1)).to(src.device)

            out = model.decode(
                memory,
                src_mask,
                generated_ids,
                tgt_mask
            )

            prob = model.generator(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)

            generated_ids = torch.cat(
                [generated_ids, next_word.unsqueeze(1)],
                dim=1
            )

            if next_word.item() == EOS:
                break

    return generated_ids


def generate_predictions(
        model,
        dataloader,
        src_sp,
        tgt_sp,
        device,
        max_len=60,
        max_samples=None,
        sample_count=5
    ):
    model.eval()

    predictions = []
    references = []
    samples = []

    if max_samples is not None:
        total = min(max_samples, len(dataloader))
    else:
        total = len(dataloader)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=total,
        desc="Generating translations"
    )

    with torch.no_grad():
        for i, (src, tgt) in progress_bar:
            if max_samples is not None and i >= max_samples:
                break

            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src)

            generated_ids = greedy_decoding(
                model=model,
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                start_symbol=BOS
            )

            src_ids = remove_special_tokens(src.squeeze(0).tolist())
            pred_ids = remove_special_tokens(generated_ids.squeeze(0).tolist())
            ref_ids = remove_special_tokens(tgt.squeeze(0).tolist())

            input_text = src_sp.decode(src_ids)
            prediction_text = tgt_sp.decode(pred_ids)
            reference_text = tgt_sp.decode(ref_ids)

            predictions.append(prediction_text)
            references.append(reference_text)

            if len(samples) < sample_count:
                samples.append({
                    "input": input_text,
                    "prediction": prediction_text,
                    "reference": reference_text
                })

            progress_bar.set_postfix({
                "samples": len(predictions)
            })

    return predictions, references, samples


def evaluate_bleu(predictions, references):
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    return bleu_score


def evaluate_chrf(predictions, references):
    chrf = CHRF()
    chrf_score = chrf.corpus_score(predictions, [references])
    return chrf_score


def evaluate_bertscore(predictions, references, device):
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        model_type="xlm-roberta-large",
        lang="en",
        device=str(device),
        verbose=True
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item()
    }


def print_sample_translations(samples):
    for i, sample in enumerate(samples):
        print(f"\n---- {i + 1}번 문장 ----")
        print("Input      :", sample["input"])
        print("Prediction :", sample["prediction"])
        print("Reference  :", sample["reference"])


def build_eval_model(mode, device):
    if mode in ["baseline", "fft"]:
        model = make_model(
            src_vocab=SRC_VOCAB_SIZE,
            tgt_vocab=TGT_VOCAB_SIZE,
            d_model=256,
            N_en=6,
            N_de=3,
            d_ff=1024,
            h=4,
            dropout=0.1,
            use_lora=False
        )

    elif mode == "lora_r4":
        model = make_model(
            src_vocab=SRC_VOCAB_SIZE,
            tgt_vocab=TGT_VOCAB_SIZE,
            d_model=256,
            N_en=6,
            N_de=3,
            d_ff=1024,
            h=4,
            dropout=0.1,
            use_lora=True,
            lora_rank=4,
            lora_alpha=16,
            lora_targets=("q", "v")
        )
    
    elif mode == "lora_r8":
        model = make_model(
            src_vocab=SRC_VOCAB_SIZE,
            tgt_vocab=TGT_VOCAB_SIZE,
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
    
    elif mode == "lora_r16":
        model = make_model(
            src_vocab=SRC_VOCAB_SIZE,
            tgt_vocab=TGT_VOCAB_SIZE,
            d_model=256,
            N_en=6,
            N_de=3,
            d_ff=1024,
            h=4,
            dropout=0.1,
            use_lora=True,
            lora_rank=16,
            lora_alpha=16,
            lora_targets=("q", "v")
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "fft", "lora_r4", "lora_r8", "lora_r16"]
    )
    args = parser.parse_args()

    mode = args.mode
    checkpoint_dir = get_save_dir(mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Mode:", mode)
    print("Checkpoint dir:", checkpoint_dir)
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    src_sp, tgt_sp = load_tokenizers(
        src_spm_path=SRC_SPM_PATH,
        tgt_spm_path=TGT_SPM_PATH
    )

    loss_loader = build_dataloader(
        parquet_path=TEST_PARQUET_PATH,
        src_spm_path=SRC_SPM_PATH,
        tgt_spm_path=TGT_SPM_PATH,
        batch_size=32,
        shuffle=False
    )

    generation_loader = build_dataloader(
        parquet_path=TEST_PARQUET_PATH,
        src_spm_path=SRC_SPM_PATH,
        tgt_spm_path=TGT_SPM_PATH,
        batch_size=1,
        shuffle=False
    )

    model = build_eval_model(
        mode=mode,
        device=device
    )

    model, checkpoint = load_model_checkpoint(
        model=model,
        save_dir=checkpoint_dir,
        device=device
    )

    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    test_loss, ppl = evaluate_loss(
        model=model,
        dataloader=loss_loader,
        device=device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")

    predictions, references, samples = generate_predictions(
        model=model,
        dataloader=generation_loader,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        device=device,
        max_len=MAX_LEN,
        max_samples=MAX_SAMPLES,
        sample_count=SAMPLE_COUNT
    )

    bleu_score = evaluate_bleu(predictions, references)
    print(f"\nBLEU: {bleu_score}")
    print(f"BLEU Score: {bleu_score.score:.4f}")

    chrf_score = evaluate_chrf(predictions, references)
    print(f"\nchrF: {chrf_score}")
    print(f"chrF Score: {chrf_score.score:.4f}")

    bert_result = evaluate_bertscore(
        predictions=predictions,
        references=references,
        device=device
    )

    print("\nBERTScore:")
    print(f"Precision: {bert_result['bertscore_precision']:.4f}")
    print(f"Recall:    {bert_result['bertscore_recall']:.4f}")
    print(f"F1:        {bert_result['bertscore_f1']:.4f}")

    print_sample_translations(samples)


if __name__ == "__main__":
    main()