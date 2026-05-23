# English-Korean Transformer Translation with LoRA

## Overview

This project implements an English-to-Korean neural machine translation model based on the Transformer architecture and applies LoRA for parameter-efficient fine-tuning.

The goal is to compare a baseline Transformer model and a LoRA-adapted Transformer in terms of translation quality and training efficiency.

## Objectives

- Implement an encoder-decoder Transformer from scratch using PyTorch
- Train the model on an English-Korean parallel corpus
- Apply LoRA to selected Transformer linear layers
- Compare baseline and LoRA models using translation quality and efficiency metrics

## Current Status

- [x] Transformer baseline implementation
- [x] Dataset preprocessing
- [x] Training loop
- [ ] Baseline evaluation
- [ ] LoRA module implementation
- [ ] LoRA training
- [ ] Baseline vs LoRA comparison
- [ ] Final report

## Project Structure

```text
enko-transformer-lora/
├── README.md
├── report.md
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
├── checkpoints/
├── results/
│   ├── baseline_result.md
│   └── lora_result.md
├── train.py
├── evaluate.py
└── .gitignore
```