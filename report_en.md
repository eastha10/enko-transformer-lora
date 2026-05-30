# Korean-English Neural Machine Translation with LoRA

## 1. Project Overview

This project implements a Korean-to-English neural machine translation model based on the Transformer architecture and applies LoRA (Low-Rank Adaptation) to compare translation quality and training efficiency.

The main goal of this project is not simply to use an existing translation model, but to directly implement the core components of the Transformer architecture using PyTorch. After building a baseline Transformer model, LoRA is applied to selected linear projection layers to examine whether translation performance can be improved or maintained with a small number of trainable parameters.

The experiment compares the following settings:

- Baseline Transformer
- Full Fine-Tuning (FFT)
- Transformer with LoRA using ranks 4, 8, and 16

---

## 2. Motivation

Transformer-based models are widely used in modern NLP tasks such as machine translation, summarization, question answering, and large language models. However, as model size increases, full fine-tuning becomes expensive because it requires updating all model parameters and consuming more GPU memory and training time.

LoRA is a parameter-efficient fine-tuning method designed to reduce this cost. Instead of updating the full weight matrix, LoRA freezes the original model weights and trains only small low-rank update matrices.

This project applies LoRA to an encoder-decoder Transformer for Korean-to-English translation and investigates the following questions:

- Can a Korean-to-English Transformer translation model be implemented from scratch using PyTorch?
- Can LoRA reduce the number of trainable parameters while maintaining or improving translation quality?
- Which Transformer components are suitable targets for LoRA adaptation?
- How do Baseline Transformer, FFT, and LoRA differ in terms of translation quality and training efficiency?

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

The baseline model is trained using teacher forcing. During training, the target sequence is shifted by one position so that the model predicts the next English token based on previous target tokens and the encoded Korean source sentence.

---

## 4. Transformer Components

### 4.1 Encoder

The encoder receives the embedded source sentence and produces contextualized representations.

Each encoder layer contains the following sublayers:

- Multi-head self-attention
- Residual connection and layer normalization
- Position-wise feed-forward network
- Residual connection and layer normalization

A source padding mask is applied during self-attention to prevent the model from attending to padding tokens.

---

### 4.2 Decoder

The decoder generates the target sentence using both previous target tokens and the encoder output.

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

where:

- `W` is the frozen original weight matrix
- `A` and `B` are trainable low-rank matrices
- `r` is the LoRA rank
- `α` is the scaling factor

In this project, the original Transformer parameters are frozen in the LoRA setting, and only the LoRA parameters are trained.

---

## 6. LoRA Target Modules

The main LoRA target modules are the linear projection layers inside multi-head attention.

Candidate target layers are:

- `W_q`: Query projection
- `W_k`: Key projection
- `W_v`: Value projection
- `W_o`: Output projection

In this experiment, LoRA is applied to:

```text
W_q and W_v
```

This setting was selected because query and value projections are commonly used LoRA targets in attention-based architectures.

---

## 7. Experimental Setup

### 7.1 Dataset

The model is designed for Korean-to-English parallel sentence translation.

```text
Source language: Korean
Target language: English
Task: Machine Translation
```

The dataset consists of Korean sentences paired with English translations.

Each sentence pair is preprocessed through the following steps:

- Tokenization
- Numericalization
- Padding
- Batching

Dataset details are as follows:

```text
Original size: approximately 1,599,972 sentence pairs
Train size: 1,519,927
Validation size: 40,046
Test size: 39,999
Source language: Korean
Target language: English
```

---

### 7.2 Tokenization

This project uses separate SentencePiece tokenizers for Korean source sentences and English target sentences.

The preprocessing pipeline includes:

- Text normalization
- Tokenization
- Vocabulary construction
- Special token insertion
- Padding
- Tensor conversion

Special tokens are defined as follows:

```text
<PAD>: Padding token
<BOS>: Beginning-of-sentence token
<EOS>: End-of-sentence token
<UNK>: Unknown token
```

The vocabulary size is set to 16,000 for both source and target tokenizers.

---

### 7.3 Model Configuration

The model configuration used in this experiment is as follows:

```text
d_model: 256
encoder layers: 6
decoder layers: 3
attention heads: 4
d_ff: 1024
dropout: 0.1
max sequence length: 128
source vocab size: 16000
target vocab size: 16000
batch size: 32
```

---

### 7.4 Training Configuration

The training configuration is as follows:

```text
Optimizer: Adam
Learning rate: 5e-5
Scheduler: None
Loss function: NLLLoss
Epochs: 5
Batch size: 32
Device: CUDA
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

The loss function ignores padding tokens so that the model is trained only on actual sentence tokens.

---

## 8. Evaluation

The model is evaluated from two perspectives:

- Translation quality
- Training efficiency

### 8.1 Translation Quality Metrics

The following metrics are used:

```text
sacreBLEU
chrF
BERTScore
```

sacreBLEU measures n-gram overlap between generated translations and reference translations.

chrF measures character-level n-gram similarity and complements BLEU by capturing surface-level similarity.

BERTScore evaluates semantic similarity using contextual embeddings.

---

### 8.2 Efficiency Metrics

To compare Baseline Transformer, FFT, and LoRA, the following efficiency metrics are considered:

```text
Total parameters
Trainable parameters
Trainable parameter ratio
Training time
GPU memory usage
Inference time
```

In the final experiment, parameter count, trainable parameter ratio, total training time, average epoch time, average samples per second, and peak GPU memory were measured for FFT and LoRA models. Baseline training time and GPU memory were not separately logged.

---

## 9. Experimental Results

This experiment compares Baseline Transformer, FFT (Full Fine-Tuning), and LoRA with ranks 4, 8, and 16 on Korean-to-English machine translation.

All models were evaluated using the epoch 5 checkpoint. Translations were generated for 39,999 test samples using greedy decoding, and BLEU, chrF, and BERTScore were computed.

### 9.1 Translation Quality

| Model | Rank | Test Loss | PPL | BLEU | chrF | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 1.6665 | 5.2936 | 27.5232 | 54.9059 | 0.9412 | 0.9380 | 0.9395 |
| FFT | - | 1.4571 | 4.2936 | 30.8565 | 58.0224 | 0.9458 | 0.9432 | 0.9445 |
| LoRA | 4 | 1.6539 | 5.2271 | 27.7273 | 55.1294 | 0.9417 | 0.9383 | 0.9399 |
| LoRA | 8 | 1.6504 | 5.2091 | 27.8064 | 55.2469 | 0.9417 | 0.9385 | 0.9400 |
| LoRA | 16 | 1.6478 | 5.1956 | 27.8453 | 55.2860 | 0.9418 | 0.9386 | 0.9401 |

The results show that FFT achieved the best performance across all evaluation metrics. FFT recorded a BLEU score of 30.8565, chrF of 58.0224, and BERTScore F1 of 0.9445.

The LoRA models performed slightly better than the Baseline Transformer. LoRA rank 4, 8, and 16 all improved BLEU, chrF, and BERTScore F1 compared with the baseline.

Among the LoRA models, rank 16 achieved the best performance. However, the improvement from rank 4 to rank 16 was limited. BLEU increased from 27.7273 to 27.8453, and BERTScore F1 increased from 0.9399 to 0.9401. This indicates that increasing the LoRA rank improved performance, but the performance gain was relatively small.

---

### 9.2 Parameter and Training Efficiency

Training efficiency was compared using trainable parameter count, total training time, average epoch time, average throughput, and peak GPU memory.  
Baseline Transformer training time and GPU memory were not separately logged, so the time and memory comparison mainly focuses on FFT and LoRA models.

| Model | Rank | Total Params | Trainable Params | Trainable Ratio | Total Training Time | Avg Epoch Time | Avg Samples/sec | Peak GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 20,203,904 | 20,203,904 | 100.0000% | - | - | - | - |
| FFT | - | 20,203,904 | 20,203,904 | 100.0000% | 68.27 min | 13.65 min | 1,855.20 | 3,104.23 MB |
| LoRA | 4 | 20,253,056 | 49,152 | 0.2427% | 61.79 min | 12.36 min | 2,049.92 | 2,476.64 MB |
| LoRA | 8 | 20,302,208 | 98,304 | 0.4842% | 62.04 min | 12.41 min | 2,043.11 | 2,477.14 MB |
| LoRA | 16 | 20,400,512 | 196,608 | 0.9637% | 62.72 min | 12.54 min | 2,020.96 | 2,484.94 MB |

LoRA trains only a very small portion of the total model parameters. For example, LoRA rank 8 trains 98,304 parameters out of 20,302,208 total parameters, corresponding to a trainable ratio of approximately 0.4842%.

LoRA also showed shorter training time than FFT. FFT required 68.27 minutes for 5 epochs, while LoRA r4, r8, and r16 required 61.79, 62.04, and 62.72 minutes, respectively. Compared with FFT, LoRA r4 was about 9.50% faster, LoRA r8 was about 9.14% faster, and LoRA r16 was about 8.13% faster.

LoRA also reduced peak GPU memory usage. FFT used approximately 3,104.23 MB of peak GPU memory, while the LoRA models used approximately 2,476 MB to 2,485 MB. This corresponds to about 20% lower peak memory usage compared with FFT.

Average throughput was also higher for LoRA. FFT processed 1,855.20 samples/sec on average, while LoRA r4, r8, and r16 processed 2,049.92, 2,043.11, and 2,020.96 samples/sec, respectively. This suggests that limiting the number of trainable parameters improves training speed and memory efficiency.

Considering both performance and efficiency, LoRA r16 achieved the best translation quality among the LoRA variants, but the performance gap between r8 and r16 was small. In contrast, r8 used half the trainable parameters of r16 and also required slightly less training time and GPU memory. Therefore, LoRA r8 can be interpreted as the best balance point in this experiment.

---

### 9.3 Training Log Comparison

The final train loss and validation loss after 5 epochs are shown below.

| Model | Rank | Final Train Loss | Final Valid Loss | Total Steps per Epoch | Total Samples per Epoch |
|---|---:|---:|---:|---:|---:|
| FFT | - | 1.6022 | 1.4492 | 47,498 | 1,519,927 |
| LoRA | 4 | 1.7995 | 1.6443 | 47,498 | 1,519,927 |
| LoRA | 8 | 1.7962 | 1.6407 | 47,498 | 1,519,927 |
| LoRA | 16 | 1.7919 | 1.6383 | 47,498 | 1,519,927 |

The training logs also show that FFT achieved the lowest train loss and validation loss. This indicates that updating all parameters allowed FFT to adapt most strongly to the training and validation data.

Among the LoRA models, final train loss and final validation loss gradually decreased as the rank increased. The final validation loss was 1.6443 for LoRA r4, 1.6407 for LoRA r8, and 1.6383 for LoRA r16. This shows that increasing rank improved training performance.

However, the validation loss difference between LoRA r8 and r16 was only 0.0024. Therefore, although LoRA r16 achieved the best validation loss among LoRA models, the improvement was limited relative to the additional number of trainable parameters.

---

### 9.4 Sample Translation Comparison

To complement the quantitative evaluation, sample translations from the Baseline Transformer, FFT, and LoRA rank 16 models are compared below. LoRA rank 16 is used as the representative LoRA model because it achieved the highest score among the LoRA variants.

| Source | Reference | Baseline Output | FFT Output | LoRA r16 Output |
|---|---|---|---|---|
| 다이제를 깔아둔 원형 틀에 반죽을 부어주세요. | Pour the dough in the round mold with Diget biscuits at the bottom. | Please bring a dough on the original frame of the diameter. | Please call the dough on the original frame that is laid off the dine. | Please bring a dough on the original frame that is placed on the diameter. |
| 괜찮은 것 같은데 이건 최대 몇 명까지 잘 수 있나요? | I think it's okay but how many people can sleep in this? | I think it's fine, so how many people can do it? | I think it's okay, how many people can do it? | I think it's fine, so how many people can do it? |
| 유럽인들과 특히 그 지배자들이 유럽대륙 안팎에 끼친 죄악은 크다. | The crimes committed by Europeans and especially by their rulers on and off continental Europe are great. | In particular, the sin of European people and its rulers in the outside of Europe. | The sinners of Europeans and their domains are particularly large in the area of Europe. | In particular, the sin of European people and its rulers in and outside Europe is largely a big sin. |
| 채팅 액션 방법은 하단의 데몬 아이콘을 클릭하면 가능한 액션들을 볼 수 있습니다. | You can check how to proceed with chatting action by clicking the demon icon below and checking available actions. | The chat action method can be seen if you click the lower-level diamond icon. | The chat action method can be seen as possible action if you click the bottom of the diamond icon. | The chat action method can be seen if you click the H ?? s Done icon. |

The examples show that all models can generate sentence-level structures to some extent, but they often fail to preserve detailed meanings. For example, in the second sentence, the phrase “잘 수 있나요?” should be translated as “can sleep,” but all models generated “can do it,” missing the key meaning.

Although FFT achieved the best average quantitative scores, it was not always the best model for each individual sample. For example, in the first sentence, FFT generated unnatural expressions such as “call the dough” and “laid off the dine.”

LoRA rank 16 produced outputs that were structurally similar to the baseline and sometimes slightly more specific. However, it also failed to correctly translate key expressions such as “Diget biscuits,” “sleep,” and “demon icon.”

Therefore, the qualitative examples suggest that FFT performs best on average, but individual sentence-level errors differ across models.

---

### 9.5 Overall Analysis

Overall, FFT achieved the highest translation quality. It recorded the best Test Loss, PPL, BLEU, chrF, and BERTScore scores. It also achieved the lowest final train loss and final validation loss in the training logs. This indicates that updating all Transformer parameters provides the strongest task adaptation.

LoRA performed worse than FFT but consistently outperformed the Baseline Transformer. LoRA rank 4, 8, and 16 all achieved higher BLEU, chrF, and BERTScore F1 than the baseline.

The LoRA rank comparison showed the following trend:

- Test Loss: r16 < r8 < r4
- PPL: r16 < r8 < r4
- BLEU: r16 > r8 > r4
- chrF: r16 > r8 > r4
- BERTScore F1: r16 > r8 > r4
- Final Valid Loss: r16 < r8 < r4

This shows that increasing the LoRA rank slightly improved performance. However, the improvement was small. From r4 to r16, BLEU increased by about 0.1180 and BERTScore F1 increased by about 0.0002.

From a parameter efficiency perspective, LoRA showed a clear advantage. LoRA r8 trained only about 0.4842% of the total parameters but still achieved better translation quality than the Baseline Transformer. It also achieved about 9.14% shorter training time and about 20% lower peak GPU memory usage than FFT.

In conclusion, FFT is the best-performing model, while LoRA is an efficient alternative that achieves better performance than the baseline with a small number of trainable parameters and lower memory usage. Among the LoRA variants, r16 achieved the best performance, but r8 is the most reasonable setting when considering the balance between performance and efficiency.

---

## 10. Discussion

This experiment compared Baseline Transformer, FFT, and LoRA rank 4/8/16 models on Korean-to-English machine translation.

The results show that FFT achieved the best translation quality. FFT recorded BLEU 30.8565, chrF 58.0224, and BERTScore F1 0.9445, achieving the highest score among all models. This indicates that updating all model parameters provides the strongest task adaptation.

LoRA did not match FFT performance, but it consistently outperformed the Baseline Transformer. In particular, LoRA r16 achieved the highest performance among LoRA variants, with BLEU 27.8453, chrF 55.2860, and BERTScore F1 0.9401.

From a parameter efficiency perspective, LoRA showed a clear advantage. LoRA r8 trained only about 0.4842% of the total parameters but still achieved better translation quality than the baseline. It also reduced training time by about 9.14% and peak GPU memory by about 20% compared with FFT.

The effect of increasing LoRA rank was observed, but the improvement was limited. BLEU, chrF, BERTScore F1, and validation loss gradually improved from LoRA r4 to r16. However, the difference between r8 and r16 was small, while r16 used twice as many trainable parameters as r8.

Therefore, if only the best LoRA performance is considered, r16 is the best choice. However, if translation quality, parameter efficiency, training time, and GPU memory are considered together, r8 is the more reasonable choice.

Overall, FFT is suitable when the goal is maximum performance, while LoRA is suitable when the goal is efficient performance improvement with limited trainable parameters and lower resource usage.

---

## 11. Limitations

This project has several limitations.

First, the experiment was conducted in a limited GPU environment using Google Colab Pro. Therefore, the model size and experimental scale are constrained compared with large-scale machine translation systems.

Second, the Baseline Transformer is a directly implemented model for educational and experimental purposes. It is not intended to compete with large pretrained translation models.

Third, Korean-to-English translation is difficult because Korean and English differ significantly in word order, morphology, particles, and sentence structure. Therefore, translation quality may be limited depending on dataset size and model capacity.

Fourth, Baseline Transformer training time and GPU memory usage were not separately logged. Therefore, the efficiency comparison for time and memory focuses mainly on FFT and LoRA models.

---

## 12. Future Work

Future improvements include:

- Applying beam search decoding
- Comparing different LoRA target modules
- Comparing attention-only LoRA and FFN-only LoRA
- Improving Korean tokenization
- Adding checkpoint-based intermediate evaluation
- Comparing LoRA with other parameter-efficient fine-tuning methods
- Extending the experiment to pretrained Transformer-based models

---

## 13. Project Structure

```text
enko-transformer-lora/
├── README.md
├── report_en.md
├── report_ko.md
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
- [x] LoRA module implementation
- [x] LoRA training
- [x] FFT training
- [x] Baseline vs LoRA vs FFT comparison
- [x] Final report

---

## 15. References

- Vaswani et al., "Attention Is All You Need"
- The Annotated Transformer
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
