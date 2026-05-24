# English-Korean Neural Machine Translation with LoRA

## 1. Project Overview

This project implements an English-to-Korean Neural Machine Translation model based on the Transformer architecture and applies LoRA, a parameter-efficient fine-tuning method, to compare translation quality and training efficiency.

The main goal of this project is not only to use an existing translation model, but to understand and implement the core components of the Transformer architecture from the model level. After building the baseline Transformer model, LoRA is applied to selected linear projection layers to analyze whether comparable translation performance can be achieved with fewer trainable parameters.

---

## 2. Motivation

Transformer-based models are widely used in modern NLP tasks, including machine translation, summarization, and large language models. However, full fine-tuning of large neural models requires substantial computational resources and memory.

LoRA addresses this problem by freezing the original model weights and training only small low-rank update matrices. This project investigates how LoRA can be applied to an encoder-decoder Transformer for English-to-Korean translation and evaluates its effect on both performance and efficiency.

The project focuses on the following questions:

- Can a Transformer-based English-to-Korean translation model be implemented from scratch using PyTorch?
- Can LoRA reduce the number of trainable parameters while maintaining translation quality?
- Which parts of the Transformer architecture are suitable targets for LoRA adaptation?
- How do baseline fine-tuning and LoRA-based fine-tuning differ in terms of performance and resource usage?

---

## 3. Baseline Model

The baseline model is an encoder-decoder Transformer implemented with PyTorch. The implementation follows the general structure of the original Transformer architecture and the Annotated Transformer implementation.

The model includes the following components:

- Token embedding
- Positional encoding
- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward network
- Residual connection
- Layer normalization
- Dropout
- Generator layer for vocabulary prediction

The overall model structure is as follows:

```text
Source Sentence
      ↓
Source Embedding + Positional Encoding
      ↓
Transformer Encoder
      ↓
Encoder Memory
      ↓
Transformer Decoder
      ↓
Linear Projection
      ↓
Log Softmax
      ↓
Target Token Prediction
```

The baseline model is trained using teacher forcing. During training, the target sequence is shifted so that the model predicts the next Korean token based on previous target tokens and the encoded English source sentence.

---

## 4. Transformer Components

### 4.1 Encoder

The encoder receives the embedded source sentence and produces contextualized representations.

Each encoder layer contains the following sublayers:

- Multi-head self-attention
- Residual connection and layer normalization
- Position-wise feed-forward network
- Residual connection and layer normalization

The source padding mask is applied during self-attention to prevent the model from attending to padding tokens.

---

### 4.2 Decoder

The decoder generates the target sentence using both the previous target tokens and the encoder output.

Each decoder layer contains the following sublayers:

- Masked multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward network
- Residual connection and layer normalization

The decoder uses a subsequent mask to prevent the model from attending to future target tokens during training.

---

### 4.3 Attention Mechanism

Scaled dot-product attention is used as the core attention mechanism.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Multi-head attention allows the model to attend to information from different representation subspaces.

In this project, attention is used in three main places:

- Encoder self-attention
- Decoder self-attention
- Encoder-decoder attention

---

## 5. LoRA Adaptation

LoRA is applied to selected linear layers inside the Transformer architecture.

Instead of updating the full weight matrix, LoRA introduces two low-rank matrices that approximate the weight update.

For an original linear layer:

```text
y = xW
```

LoRA modifies it as:

```text
y = xW + xBA * α/r
```

Where:

- `W` is the frozen original weight matrix
- `A` and `B` are trainable low-rank matrices
- `r` is the LoRA rank
- `α` is the scaling factor

In this project, the original Transformer parameters are frozen, and only the LoRA parameters are trained.

---

## 6. LoRA Target Modules

The main LoRA target modules are the linear projection layers in multi-head attention.

Candidate target layers:

- `W_q`: Query projection
- `W_k`: Key projection
- `W_v`: Value projection
- `W_o`: Output projection

Primary target setting:

```text
W_q and W_v
```

Additional experiments may include:

```text
W_q + W_k + W_v + W_o
FFN linear layers
Attention + FFN combined setting
```

The purpose of these experiments is to compare how LoRA placement affects translation quality and parameter efficiency.

---

## 7. Experimental Setup

### 7.1 Dataset

The model is designed for English-to-Korean parallel sentence translation.

```text
Source language: English
Target language: Korean
Task: Machine Translation
```

The dataset consists of English sentences paired with Korean translations.

Each sentence pair is preprocessed through tokenization, numericalization, padding, and batching.

Dataset details will be updated after final preprocessing.

```text
Dataset:
Train size:
Validation size:
Test size:
Maximum sequence length:
```

---

### 7.2 Tokenization

The project supports tokenizer-based preprocessing for both English and Korean.

The preprocessing pipeline includes:

- Text normalization
- Tokenization
- Vocabulary construction
- Special token insertion
- Padding
- Tensor conversion

Special tokens:

```text
<PAD>: Padding token
<BOS>: Beginning-of-sentence token
<EOS>: End-of-sentence token
<UNK>: Unknown token
```

---

### 7.3 Model Configuration

Initial baseline configuration:

```text
d_model: 256
encoder layers: 6
decoder layers: 3
attention heads: 4
d_ff: 1024
dropout: 0.1
max sequence length: 128
vocab size: 16000
batch size: 32 or 64
```

Expanded configuration if the baseline training is stable:

```text
d_model: 512
encoder layers: 12
decoder layers: 6
attention heads: 8
d_ff: 2048
vocab size: 32000
batch size: 32 to 128
```

---

### 7.4 Training Configuration

```text
Optimizer:
Learning rate:
Scheduler:
Loss function:
Epochs:
Batch size:
Device:
GPU:
```

The loss function ignores padding tokens so that the model is not penalized for predicting padded positions.

---

## 8. Evaluation

The model is evaluated from two perspectives:

- Translation quality
- Training efficiency

### 8.1 Translation Quality Metrics

The following metrics are used:

```text
sacreBLEU
BERTScore
```

sacreBLEU is used to measure n-gram overlap between generated translations and reference translations.

BERTScore is used to evaluate semantic similarity using contextual embeddings.

---

### 8.2 Efficiency Metrics

To compare baseline training and LoRA-based training, the following efficiency metrics are measured:

```text
Total parameters
Trainable parameters
Trainable parameter ratio
Training time
GPU memory usage
Inference time
```

This allows the project to evaluate not only whether LoRA maintains translation quality, but also whether it improves training efficiency.

---

## 9. Results

Final results will be summarized in the following format.

### 9.1 Translation Quality

| Model | BLEU | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|---|---:|---:|---:|---:|
| Baseline Transformer | - | - | - | - |
| Transformer + LoRA | - | - | - | - |

---

### 9.2 Efficiency Comparison

| Model | Total Params | Trainable Params | Trainable Ratio | Training Time | GPU Memory |
|---|---:|---:|---:|---:|---:|
| Baseline Transformer | - | - | - | - | - |
| Transformer + LoRA | - | - | - | - | - |

---

### 9.3 Sample Translation Results

| Source | Reference | Baseline Output | LoRA Output |
|---|---|---|---|
| - | - | - | - |
| - | - | - | - |
| - | - | - | - |

---

## 10. Discussion

This project compares full Transformer training and LoRA-based parameter-efficient training in an English-to-Korean translation task.

The baseline model updates all trainable parameters, which may provide stronger task adaptation but requires more computation.

In contrast, LoRA freezes the original model weights and trains only a small number of additional parameters. Therefore, LoRA is expected to reduce trainable parameter count and memory usage.

The main analysis focuses on the following points:

- Whether LoRA can maintain translation quality with fewer trainable parameters
- How much the trainable parameter count is reduced
- Whether LoRA reduces GPU memory usage during training
- Whether attention projection layers are effective LoRA targets
- Whether FFN layers provide additional gains when adapted with LoRA

---

## 11. Limitations

This project has several limitations.

First, the model is trained under limited computational resources using Google Colab Pro. Therefore, the model size and dataset size are constrained compared to large-scale machine translation systems.

Second, the baseline Transformer is implemented for educational and experimental purposes. It is not intended to compete with large pretrained translation models.

Third, English-to-Korean translation is a difficult task due to differences in word order, morphology, and sentence structure. Therefore, translation quality may be limited when the dataset size is small.

---

## 12. Future Work

Future improvements include:

- Applying beam search decoding
- Testing different LoRA ranks
- Applying LoRA to different target modules
- Comparing attention-only LoRA and FFN-only LoRA
- Using a larger English-Korean parallel corpus
- Improving Korean tokenization
- Adding checkpoint-based evaluation
- Comparing LoRA with other parameter-efficient fine-tuning methods
- Extending the experiment to pretrained Transformer-based models

---

## 13. Project Structure

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

---

## 14. Current Status

- [x] Transformer baseline implementation
- [x] Dataset preprocessing
- [x] Training loop
- [x] Baseline evaluation
- [ ] LoRA module implementation
- [ ] LoRA training
- [ ] Baseline vs LoRA comparison
- [ ] Final report

---

## 15. References

- Vaswani et al., "Attention Is All You Need"
- The Annotated Transformer
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
