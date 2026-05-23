# LoRA 기반 영어-한국어 Transformer 기계번역 프로젝트 보고서

## 1. 프로젝트 개요

이 프로젝트는 Transformer 구조를 기반으로 영어-한국어 기계번역 모델을 직접 구현하고, LoRA(Low-Rank Adaptation)를 적용하여 번역 품질과 학습 효율성을 비교하는 것을 목표로 합니다.

이 프로젝트의 핵심 목적은 단순히 기존 번역 모델을 사용하는 것이 아니라, Transformer의 주요 구성 요소를 PyTorch 기반으로 직접 구현하고, 이후 LoRA를 적용하여 적은 수의 학습 파라미터만으로도 번역 성능을 유지할 수 있는지 확인하는 것입니다.

Baseline Transformer 모델을 먼저 구현한 뒤, 선택된 Linear projection layer에 LoRA를 적용하여 baseline 학습 방식과 LoRA 기반 학습 방식을 비교합니다.

---

## 2. 연구 동기

Transformer 기반 모델은 기계번역, 문서 요약, 질의응답, 대규모 언어 모델 등 현대 NLP 분야에서 핵심적인 구조로 사용됩니다. 하지만 모델 규모가 커질수록 전체 파라미터를 학습하는 full fine-tuning 방식은 많은 GPU 메모리와 학습 시간을 요구합니다.

LoRA는 이러한 문제를 완화하기 위한 parameter-efficient fine-tuning 방법입니다. 기존 모델의 weight는 고정하고, 작은 low-rank matrix만 추가로 학습하여 전체 학습 비용을 줄이는 방식입니다.

이 프로젝트에서는 LoRA를 Encoder-Decoder Transformer 기반 영어-한국어 번역 모델에 적용하여 다음 질문을 실험적으로 확인하고자 합니다.

- PyTorch를 사용하여 영어-한국어 Transformer 번역 모델을 직접 구현할 수 있는가?
- LoRA를 적용했을 때 학습 가능한 파라미터 수를 줄이면서 번역 품질을 유지할 수 있는가?
- Transformer 내부에서 LoRA를 적용하기 적절한 layer는 어디인가?
- Baseline 학습과 LoRA 기반 학습은 성능 및 자원 사용량 측면에서 어떤 차이를 보이는가?

---

## 3. Baseline 모델

Baseline 모델은 PyTorch로 구현한 Encoder-Decoder Transformer 구조를 사용합니다. 전체 구조는 원 논문인 "Attention Is All You Need"와 Annotated Transformer 구현을 참고합니다.

모델은 다음 구성 요소를 포함합니다.

- Token embedding
- Positional encoding
- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward network
- Residual connection
- Layer normalization
- Dropout
- Vocabulary prediction을 위한 Generator layer

전체 모델 흐름은 다음과 같습니다.

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

Baseline 모델은 teacher forcing 방식으로 학습합니다. 학습 과정에서는 target sequence를 한 칸 shift하여, 모델이 이전 target token들과 encoder output을 기반으로 다음 한국어 token을 예측하도록 합니다.

---

## 4. Transformer 구성 요소

### 4.1 Encoder

Encoder는 embedding된 source sentence를 입력으로 받아 contextualized representation을 생성합니다.

각 encoder layer는 다음 sublayer들로 구성됩니다.

- Multi-head self-attention
- Residual connection and layer normalization
- Position-wise feed-forward network
- Residual connection and layer normalization

Self-attention 과정에서는 source padding mask를 사용하여 padding token에 attention이 가지 않도록 합니다.

---

### 4.2 Decoder

Decoder는 이전 target token들과 encoder output을 함께 사용하여 target sentence를 생성합니다.

각 decoder layer는 다음 sublayer들로 구성됩니다.

- Masked multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward network
- Residual connection and layer normalization

Decoder의 masked self-attention에서는 subsequent mask를 사용합니다. 이는 학습 중 모델이 현재 위치보다 뒤에 있는 미래 target token을 참조하지 못하게 하기 위한 장치입니다.

---

### 4.3 Attention Mechanism

이 프로젝트에서는 scaled dot-product attention을 핵심 attention mechanism으로 사용합니다.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Multi-head attention은 서로 다른 representation subspace에서 정보를 병렬적으로 참조할 수 있게 합니다.

이 프로젝트에서 attention은 크게 세 위치에서 사용됩니다.

- Encoder self-attention
- Decoder self-attention
- Encoder-decoder attention

---

## 5. LoRA 적용 방식

LoRA는 Transformer 내부의 선택된 Linear layer에 적용합니다.

기존 full fine-tuning 방식은 전체 weight matrix를 직접 업데이트합니다. 반면 LoRA는 원래 weight를 freeze하고, weight update를 두 개의 작은 low-rank matrix로 근사합니다.

기존 Linear layer가 다음과 같다고 할 때,

```text
y = xW
```

LoRA 적용 후에는 다음과 같은 형태가 됩니다.

```text
y = xW + xBA * α/r
```

각 기호의 의미는 다음과 같습니다.

- `W`: freeze된 기존 weight matrix
- `A`, `B`: 학습 가능한 low-rank matrix
- `r`: LoRA rank
- `α`: scaling factor

이 프로젝트에서는 기존 Transformer parameter를 freeze하고, LoRA parameter만 학습하도록 설정합니다.

---

## 6. LoRA 적용 대상 모듈

LoRA의 주요 적용 대상은 multi-head attention 내부의 Linear projection layer입니다.

후보 layer는 다음과 같습니다.

- `W_q`: Query projection
- `W_k`: Key projection
- `W_v`: Value projection
- `W_o`: Output projection

초기 실험에서는 다음 설정을 우선 적용합니다.

```text
W_q and W_v
```

추가 실험에서는 다음 설정도 비교할 수 있습니다.

```text
W_q + W_k + W_v + W_o
FFN linear layers
Attention + FFN combined setting
```

이 실험의 목적은 LoRA 적용 위치가 번역 품질과 파라미터 효율성에 어떤 영향을 주는지 비교하는 것입니다.

---

## 7. 실험 설정

### 7.1 Dataset

모델은 영어-한국어 병렬 문장 번역을 대상으로 설계합니다.

```text
Source language: English
Target language: Korean
Task: Machine Translation
```

Dataset은 영어 문장과 한국어 번역 문장의 쌍으로 구성됩니다.

각 문장 쌍은 다음 전처리 과정을 거칩니다.

- Tokenization
- Numericalization
- Padding
- Batching

Dataset 세부 정보는 최종 전처리 이후 업데이트합니다.

```text
Dataset:
Train size:
Validation size:
Test size:
Maximum sequence length:
```

---

### 7.2 Tokenization

이 프로젝트는 영어와 한국어 각각에 대해 tokenizer 기반 전처리를 수행합니다.

전처리 pipeline은 다음 단계로 구성됩니다.

- Text normalization
- Tokenization
- Vocabulary construction
- Special token insertion
- Padding
- Tensor conversion

사용하는 special token은 다음과 같습니다.

```text
<PAD>: Padding token
<BOS>: Beginning-of-sentence token
<EOS>: End-of-sentence token
<UNK>: Unknown token
```

---

### 7.3 Model Configuration

초기 baseline configuration은 다음과 같습니다.

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

Baseline 학습이 안정적으로 진행될 경우 다음 확장 configuration을 사용할 계획입니다.

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

학습 설정은 다음 항목을 기준으로 정리합니다.

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

Loss function은 padding token에 대한 loss를 계산하지 않도록 설정합니다. 이를 통해 모델이 실제 문장 token에 대해서만 학습하도록 합니다.

---

## 8. 평가 방법

모델 평가는 크게 두 관점에서 진행합니다.

- 번역 품질
- 학습 효율성

### 8.1 번역 품질 평가 지표

번역 품질 평가는 다음 지표를 사용합니다.

```text
sacreBLEU
BERTScore
```

sacreBLEU는 생성된 번역문과 reference 번역문 사이의 n-gram overlap을 측정합니다.

BERTScore는 contextual embedding을 사용하여 생성 문장과 reference 문장의 의미적 유사도를 평가합니다.

---

### 8.2 학습 효율성 평가 지표

Baseline 학습과 LoRA 기반 학습을 비교하기 위해 다음 효율성 지표를 측정합니다.

```text
Total parameters
Trainable parameters
Trainable parameter ratio
Training time
GPU memory usage
Inference time
```

이를 통해 LoRA가 번역 성능을 유지하는지뿐만 아니라, 실제로 학습 자원 사용량을 줄이는지도 확인합니다.

---

## 9. 실험 결과

최종 실험 결과는 다음 형식으로 정리할 예정입니다.

### 9.1 번역 품질 비교

| Model | BLEU | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|---|---:|---:|---:|---:|
| Baseline Transformer | - | - | - | - |
| Transformer + LoRA | - | - | - | - |

---

### 9.2 학습 효율성 비교

| Model | Total Params | Trainable Params | Trainable Ratio | Training Time | GPU Memory |
|---|---:|---:|---:|---:|---:|
| Baseline Transformer | - | - | - | - | - |
| Transformer + LoRA | - | - | - | - | - |

---

### 9.3 번역 예시 비교

| Source | Reference | Baseline Output | LoRA Output |
|---|---|---|---|
| - | - | - | - |
| - | - | - | - |
| - | - | - | - |

---

## 10. 분석 및 논의

이 프로젝트는 영어-한국어 번역 task에서 full Transformer 학습 방식과 LoRA 기반 parameter-efficient 학습 방식을 비교합니다.

Baseline 모델은 전체 trainable parameter를 업데이트하기 때문에 task adaptation 측면에서 강점을 가질 수 있습니다. 하지만 그만큼 학습 시간, GPU memory, 저장 공간 측면에서 비용이 커질 수 있습니다.

반면 LoRA는 기존 weight를 고정하고 소수의 추가 parameter만 학습합니다. 따라서 trainable parameter 수와 GPU memory 사용량을 줄일 수 있을 것으로 기대됩니다.

주요 분석 관점은 다음과 같습니다.

- LoRA가 적은 trainable parameter만으로 번역 품질을 유지할 수 있는가?
- Trainable parameter 수가 baseline 대비 얼마나 감소하는가?
- LoRA 적용 시 GPU memory 사용량이 감소하는가?
- Attention projection layer가 LoRA 적용 대상으로 효과적인가?
- FFN layer까지 LoRA를 확장했을 때 추가적인 성능 향상이 있는가?

---

## 11. 한계점

이 프로젝트에는 다음과 같은 한계가 있습니다.

첫째, 실험은 Google Colab Pro의 제한된 GPU 환경에서 수행됩니다. 따라서 대규모 기계번역 시스템과 비교했을 때 모델 크기와 dataset 크기에 제약이 있습니다.

둘째, baseline Transformer는 학습 및 실험 목적의 직접 구현 모델입니다. 따라서 대규모 pretrained translation model과 직접적으로 경쟁하는 것을 목표로 하지 않습니다.

셋째, 영어-한국어 번역은 어순, 형태소, 조사, 문장 구조 차이가 크기 때문에 dataset 크기가 작을 경우 번역 품질에 한계가 있을 수 있습니다.

---

## 12. 향후 개선 방향

향후 개선할 수 있는 부분은 다음과 같습니다.

- Beam search decoding 적용
- LoRA rank 변화 실험
- LoRA target module별 비교 실험
- Attention-only LoRA와 FFN-only LoRA 비교
- 더 큰 영어-한국어 병렬 corpus 사용
- 한국어 tokenization 방식 개선
- Checkpoint 기반 중간 평가 추가
- LoRA 외 다른 parameter-efficient fine-tuning 방법과 비교
- Pretrained Transformer 기반 모델로 확장

---

## 13. 프로젝트 파일 구조

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

## 14. 현재 진행 상태

- [x] Transformer baseline implementation
- [x] Dataset preprocessing
- [x] Training loop
- [ ] Baseline evaluation
- [ ] LoRA module implementation
- [ ] LoRA training
- [ ] Baseline vs LoRA comparison
- [ ] Final report

---

## 15. 참고 자료

- Vaswani et al., "Attention Is All You Need"
- The Annotated Transformer
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
