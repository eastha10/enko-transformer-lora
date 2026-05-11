# English-Korean Neural Machine Translation with LoRA

## 1. Project Overview

This project implements an English-to-Korean Neural Machine Translation model based on the Transformer architecture and applies LoRA, a parameter-efficient fine-tuning method, to compare translation quality and training efficiency.

The main goal of this project is not only to use an existing translation model, but to understand and implement the core components of the Transformer architecture from the model level. After building the baseline Transformer model, LoRA is applied to selected linear projection layers to analyze whether comparable translation performance can be achieved with fewer trainable parameters.

---

## 2. Motivation

Transformer-based models are widely used in modern NLP tasks, including machine translation, summarization, and large language models. However, full fine-tuning of large neural models requires substantial computational resources and memory.

LoRA addresses this problem by freezing the original model weights and training only small low-rank update matrices. This project investigates how LoRA can be applied to an encoder-decoder Transformer for English-to-Korean translation and evaluates its effect on both performance and efficiency.

The project focuses on the following questions:

1. Can a Transformer-based English-to-Korean translation model be implemented from scratch using PyTorch?
2. Can LoRA reduce the number of trainable parameters while maintaining translation quality?
3. Which parts of the Transformer architecture are suitable targets for LoRA adaptation?
4. How do baseline fine-tuning and LoRA-based fine-tuning differ in terms of performance and resource usage?

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