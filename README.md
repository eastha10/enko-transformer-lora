# Korean-English Transformer Translation with LoRA

This repository implements a Korean-to-English neural machine translation system based on the Transformer architecture and compares full fine-tuning with LoRA-based parameter-efficient fine-tuning.

The project focuses on implementing the Transformer model directly in PyTorch, training it on a Korean-English parallel corpus, applying LoRA to selected attention projection layers, and evaluating translation quality and training efficiency.

---

## Overview

This project compares the following model settings:

- **Baseline Transformer**: encoder-decoder Transformer trained as the base model
- **FFT**: full fine-tuning with all Transformer parameters trainable
- **LoRA r4 / r8 / r16**: LoRA applied to selected attention projection layers with ranks 4, 8, and 16

The translation direction is:

```text
Korean → English
```

---

## Objectives

- Implement an encoder-decoder Transformer from scratch using PyTorch
- Preprocess a Korean-English parallel corpus with SentencePiece tokenizers
- Train and evaluate a baseline Transformer translation model
- Apply LoRA to selected Transformer attention projection layers
- Compare Baseline, FFT, and LoRA models using translation quality metrics
- Compare parameter efficiency, training time, throughput, and GPU memory usage

---

## Model Architecture

The baseline model follows the standard encoder-decoder Transformer architecture.

Main components:

- Token embedding
- Positional encoding
- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward network
- Residual connection
- Layer normalization
- Dropout
- Generator layer with log-softmax output

Model configuration:

```text
d_model: 256
encoder layers: 6
decoder layers: 3
attention heads: 4
d_ff: 1024
dropout: 0.1
source vocab size: 16000
target vocab size: 16000
batch size: 32
```

---

## LoRA Configuration

LoRA is applied to the linear projection layers inside multi-head attention.

Target modules:

```text
W_q and W_v
```

Tested LoRA ranks:

```text
r = 4, 8, 16
```

In the LoRA setting, the original Transformer parameters are frozen and only the LoRA parameters are trained.

---

## Dataset

The dataset consists of Korean-English parallel sentence pairs.

```text
Original size: approximately 1,599,972 sentence pairs
Train size: 1,519,927
Validation size: 40,046
Test size: 39,999
Source language: Korean
Target language: English
```

The dataset is tokenized with separate SentencePiece tokenizers for source and target languages.

Special token IDs:

```text
PAD = 0
UNK = 1
BOS = 2
EOS = 3
```

---

## Experimental Results

All models were evaluated on the test set using greedy decoding.

### Translation Quality

| Model | Rank | Test Loss | PPL | BLEU | chrF | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 1.6665 | 5.2936 | 27.5232 | 54.9059 | 0.9412 | 0.9380 | 0.9395 |
| FFT | - | 1.4571 | 4.2936 | 30.8565 | 58.0224 | 0.9458 | 0.9432 | 0.9445 |
| LoRA | 4 | 1.6539 | 5.2271 | 27.7273 | 55.1294 | 0.9417 | 0.9383 | 0.9399 |
| LoRA | 8 | 1.6504 | 5.2091 | 27.8064 | 55.2469 | 0.9417 | 0.9385 | 0.9400 |
| LoRA | 16 | 1.6478 | 5.1956 | 27.8453 | 55.2860 | 0.9418 | 0.9386 | 0.9401 |

### Training Efficiency

| Model | Rank | Total Params | Trainable Params | Trainable Ratio | Total Training Time | Avg Epoch Time | Avg Samples/sec | Peak GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 20,203,904 | 20,203,904 | 100.0000% | - | - | - | - |
| FFT | - | 20,203,904 | 20,203,904 | 100.0000% | 68.27 min | 13.65 min | 1,855.20 | 3,104.23 MB |
| LoRA | 4 | 20,253,056 | 49,152 | 0.2427% | 61.79 min | 12.36 min | 2,049.92 | 2,476.64 MB |
| LoRA | 8 | 20,302,208 | 98,304 | 0.4842% | 62.04 min | 12.41 min | 2,043.11 | 2,477.14 MB |
| LoRA | 16 | 20,400,512 | 196,608 | 0.9637% | 62.72 min | 12.54 min | 2,020.96 | 2,484.94 MB |

### Training Log Summary

| Model | Rank | Final Train Loss | Final Valid Loss | Steps per Epoch | Samples per Epoch |
|---|---:|---:|---:|---:|---:|
| FFT | - | 1.6022 | 1.4492 | 47,498 | 1,519,927 |
| LoRA | 4 | 1.7995 | 1.6443 | 47,498 | 1,519,927 |
| LoRA | 8 | 1.7962 | 1.6407 | 47,498 | 1,519,927 |
| LoRA | 16 | 1.7919 | 1.6383 | 47,498 | 1,519,927 |

---

## Key Findings

- FFT achieved the best translation quality across all metrics.
- LoRA models consistently outperformed the Baseline Transformer.
- LoRA rank 16 achieved the best performance among LoRA variants.
- LoRA rank 8 provided the best balance between performance and efficiency.
- LoRA rank 8 trained only 0.4842% of the total parameters.
- LoRA rank 8 reduced training time by about 9.14% compared with FFT.
- LoRA reduced peak GPU memory usage by about 20% compared with FFT.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/eastha10/enko-transformer-lora.git
cd enko-transformer-lora
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data and Checkpoints

Large data and checkpoint files are not tracked by Git.

Expected local paths:

```text
data/sampled/train.parquet
data/sampled/valid.parquet
data/sampled/test.parquet
tokenizer/src_spm.model
tokenizer/tgt_spm.model
checkpoints/checkpoint-epoch-*.pt
```

The following file types and directories should remain excluded from Git:

```text
data/
checkpoints/
*.pt
*.pth
*.ckpt
*.parquet
__pycache__/
*.pyc
```

---

## Usage

Train the baseline or fine-tuning models according to the mode supported by `train.py`.

Example commands:

```bash
python train.py --mode baseline
python train.py --mode fft
python train.py --mode lora --rank 4
python train.py --mode lora --rank 8
python train.py --mode lora --rank 16
```

Run evaluation:

```bash
python evaluate.py --mode baseline
python evaluate.py --mode fft
python evaluate.py --mode lora --rank 4
python evaluate.py --mode lora --rank 8
python evaluate.py --mode lora --rank 16
```

If the local script uses different argument names, adjust the command-line flags to match the current `argparse` configuration.

---

## Project Structure

```text
enko-transformer-lora/
├── README.md
├── report_ko.md
├── report_en.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   └── lora.yaml
├── notebooks/
│   └── experiments.ipynb
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── modules.py
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── lora.py
│   ├── data.py
│   ├── checkpoint.py
│   └── inference.py
├── train.py
├── evaluate.py
└── .gitignore
```

---

## Current Status

- [x] Transformer baseline implementation
- [x] Dataset preprocessing
- [x] SentencePiece tokenizer training
- [x] Training loop
- [x] Baseline evaluation
- [x] LoRA module implementation
- [x] LoRA training
- [x] FFT training
- [x] BLEU, chrF, and BERTScore evaluation
- [x] Training efficiency comparison
- [x] Korean report
- [x] English report

---

## Limitations

- The model is a directly implemented Transformer for experimental purposes.
- It is not intended to compete with large pretrained translation models.
- Training was conducted under limited GPU resources.
- Training time and GPU memory were logged for FFT and LoRA models, but not for the original Baseline Transformer.
- Greedy decoding was used for evaluation, so beam search may improve translation quality.

---

## Future Work

- Apply beam search decoding
- Compare different LoRA target modules
- Compare attention-only LoRA and FFN-only LoRA
- Improve Korean tokenization
- Add checkpoint-based intermediate evaluation
- Compare LoRA with other parameter-efficient fine-tuning methods
- Extend the experiment to pretrained Transformer-based models

---

## References

- Vaswani et al., "Attention Is All You Need"
- The Annotated Transformer
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
