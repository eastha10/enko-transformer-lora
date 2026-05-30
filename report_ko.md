# LoRA 기반 한국어-영어 Transformer 기계번역 프로젝트 보고서

## 1. 프로젝트 개요

이 프로젝트는 Transformer 구조를 기반으로 한국어-영어 기계번역 모델을 직접 구현하고, LoRA(Low-Rank Adaptation)를 적용하여 번역 품질과 학습 효율성을 비교하는 것을 목표로 합니다.

이 프로젝트의 핵심 목적은 단순히 기존 번역 모델을 사용하는 것이 아니라, Transformer의 주요 구성 요소를 PyTorch 기반으로 직접 구현하고, 이후 LoRA를 적용하여 적은 수의 학습 파라미터만으로도 번역 성능을 유지할 수 있는지 확인하는 것입니다.

Baseline Transformer 모델을 먼저 구현한 뒤, 선택된 Linear projection layer에 LoRA를 적용하여 baseline 학습 방식과 LoRA 기반 학습 방식을 비교합니다.

---

## 2. 연구 동기

Transformer 기반 모델은 기계번역, 문서 요약, 질의응답, 대규모 언어 모델 등 현대 NLP 분야에서 핵심적인 구조로 사용됩니다. 하지만 모델 규모가 커질수록 전체 파라미터를 학습하는 full fine-tuning 방식은 많은 GPU 메모리와 학습 시간을 요구합니다.

LoRA는 이러한 문제를 완화하기 위한 parameter-efficient fine-tuning 방법입니다. 기존 모델의 weight는 고정하고, 작은 low-rank matrix만 추가로 학습하여 전체 학습 비용을 줄이는 방식입니다.

이 프로젝트에서는 LoRA를 Encoder-Decoder Transformer 기반 한국어-영어 번역 모델에 적용하여 다음 질문을 실험적으로 확인하고자 합니다.

- PyTorch를 사용하여 한국어-영어 Transformer 번역 모델을 직접 구현할 수 있는가?
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

Baseline 모델은 teacher forcing 방식으로 학습합니다. 학습 과정에서는 target sequence를 한 칸 shift하여, 모델이 이전 target token들과 encoder output을 기반으로 다음 영어 token을 예측하도록 합니다.

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

이 실험에서는 다음 설정을 우선 적용합니다.

```text
W_q and W_v
```

---

## 7. 실험 설정

### 7.1 Dataset

모델은 한국어-영어 병렬 문장 번역을 대상으로 설계합니다.

```text
Source language: Korean
Target language: English
Task: Machine Translation
```

Dataset은 한국어 문장과 영어 번역 문장의 쌍으로 구성됩니다.

각 문장 쌍은 다음 전처리 과정을 거칩니다.

- Tokenization
- Numericalization
- Padding
- Batching

```text
Dataset:
- Original size: 약 1,599,972 sentence pairs
- Train size: 1,519,927
- Validation size: 40,046
- Test size: 39,999
- Source language: Korean
- Target language: English
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
batch size: 32
```

---

### 7.4 Training Configuration

학습 설정은 다음 항목을 기준으로 정리합니다.

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
chrF
BERTScore
```

sacreBLEU는 생성된 번역문과 reference 번역문 사이의 n-gram overlap을 측정합니다.

BERTScore는 contextual embedding을 사용하여 생성 문장과 reference 문장의 의미적 유사도를 평가합니다.

chrF는 문자 기반 n-gram 유사도를 사용하여 BLEU가 포착하지 못하는 표면적 유사도를 보완적으로 평가합니다.

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

본 실험은 한국어-영어 번역 태스크에서 기본 Transformer, FFT(Full Fine-Tuning), LoRA rank 4/8/16 모델의 번역 품질과 파라미터 효율성을 비교하였다.  
모든 모델은 epoch 5 checkpoint를 기준으로 평가하였으며, test set 39,999개 문장에 대해 greedy decoding으로 번역을 생성한 뒤 BLEU, chrF, BERTScore를 계산하였다.

### 9.1 번역 품질 비교

| Model | Rank | Test Loss | PPL | BLEU | chrF | BERTScore - Precision | BERTScore - Recall | BERTScore - F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 1.6665 | 5.2936 | 27.5232 | 54.9059 | 0.9412 | 0.9380 | 0.9395 |
| FFT | - | 1.4571 | 4.2936 | 30.8565 | 58.0224 | 0.9458 | 0.9432 | 0.9445 |
| LoRA | 4 | 1.6539 | 5.2271 | 27.7273 | 55.1294 | 0.9417 | 0.9383 | 0.9399 |
| LoRA | 8 | 1.6504 | 5.2091 | 27.8064 | 55.2469 | 0.9417 | 0.9385 | 0.9400 |
| LoRA | 16 | 1.6478 | 5.1956 | 27.8453 | 55.2860 | 0.9418 | 0.9386 | 0.9401 |

평가 결과, FFT가 모든 지표에서 가장 높은 성능을 보였다. FFT는 BLEU 30.8565, chrF 58.0224, BERTScore F1 0.9445를 기록하여 Baseline과 LoRA 계열 모델보다 우수하였다.

LoRA 계열 모델은 Baseline Transformer보다 소폭 높은 성능을 보였다. LoRA rank 4, 8, 16 모두 Baseline보다 BLEU, chrF, BERTScore F1이 개선되었다. 특히 rank가 증가할수록 Test Loss, PPL, BLEU, chrF, BERTScore F1이 모두 조금씩 개선되는 경향을 보였다.

다만 LoRA rank 증가에 따른 성능 향상 폭은 크지 않았다. rank 4에서 rank 16으로 증가했을 때 BLEU는 27.7273에서 27.8453으로 약 0.1180 상승하였고, BERTScore F1은 0.9399에서 0.9401로 약 0.0002 상승하였다. 따라서 본 실험에서는 LoRA rank 증가가 성능 개선으로 이어지기는 했지만, 개선 폭은 제한적이었다.

---

### 9.2 파라미터 및 학습 효율성 비교

학습 효율성은 각 모델의 trainable parameter 수, 전체 학습 시간, epoch당 평균 학습 시간, 평균 처리 속도, peak GPU memory를 기준으로 비교하였다.  
Baseline Transformer의 학습 시간 및 GPU memory 로그는 별도로 저장하지 않았으므로, 시간·메모리 비교는 FFT와 LoRA 계열 모델을 중심으로 분석하였다.

| Model | Rank | Total Params | Trainable Params | Trainable Ratio | Total Training Time | Avg Epoch Time | Avg Samples/sec | Peak GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Transformer | - | 20,203,904 | 20,203,904 | 100.0000% | - | - | - | - |
| FFT | - | 20,203,904 | 20,203,904 | 100.0000% | 68.27 min | 13.65 min | 1,855.20 | 3,104.23 MB |
| LoRA | 4 | 20,253,056 | 49,152 | 0.2427% | 61.79 min | 12.36 min | 2,049.92 | 2,476.64 MB |
| LoRA | 8 | 20,302,208 | 98,304 | 0.4842% | 62.04 min | 12.41 min | 2,043.11 | 2,477.14 MB |
| LoRA | 16 | 20,400,512 | 196,608 | 0.9637% | 62.72 min | 12.54 min | 2,020.96 | 2,484.94 MB |

LoRA는 전체 모델 파라미터 중 매우 적은 수의 파라미터만 학습하도록 제한하였다. LoRA rank 8 기준 trainable parameter는 98,304개이며, 전체 20,302,208개 파라미터 중 약 0.4842%만 학습하였다.

학습 시간 측면에서도 LoRA는 FFT보다 짧은 학습 시간을 보였다. FFT는 5 epoch 학습에 총 68.27분이 소요되었고, LoRA r4, r8, r16은 각각 61.79분, 62.04분, 62.72분이 소요되었다. FFT와 비교했을 때 LoRA r4는 약 9.50%, LoRA r8은 약 9.14%, LoRA r16은 약 8.13% 짧은 학습 시간을 기록하였다.

GPU memory 사용량에서도 LoRA의 효율성이 확인되었다. FFT의 peak GPU memory는 약 3,104.23MB였고, LoRA 계열 모델은 약 2,476MB에서 2,485MB 수준을 기록하였다. 이는 FFT 대비 약 20% 낮은 peak memory 사용량이다.

평균 처리 속도 역시 LoRA가 FFT보다 높았다. FFT는 평균 1,855.20 samples/sec를 기록한 반면, LoRA r4, r8, r16은 각각 2,049.92, 2,043.11, 2,020.96 samples/sec를 기록하였다. 이는 LoRA가 학습 대상 파라미터를 제한함으로써 학습 속도와 메모리 사용량 측면에서 더 효율적임을 보여준다.

성능과 효율을 함께 고려하면, LoRA r16은 LoRA 계열 중 가장 높은 번역 품질을 보였지만 r8과의 성능 차이는 작았다. 반면 r8은 r16보다 trainable parameter 수가 절반이며, 학습 시간과 GPU memory도 더 낮았다. 따라서 본 실험에서는 LoRA r8이 성능과 효율의 균형점으로 해석될 수 있다.

---

### 9.3 학습 로그 비교

각 모델의 5 epoch 학습 후 최종 train loss와 validation loss는 다음과 같다.

| Model | Rank | Final Train Loss | Final Valid Loss | Total Steps per Epoch | Total Samples per Epoch |
|---|---:|---:|---:|---:|---:|
| FFT | - | 1.6022 | 1.4492 | 47,498 | 1,519,927 |
| LoRA | 4 | 1.7995 | 1.6443 | 47,498 | 1,519,927 |
| LoRA | 8 | 1.7962 | 1.6407 | 47,498 | 1,519,927 |
| LoRA | 16 | 1.7919 | 1.6383 | 47,498 | 1,519,927 |

학습 로그 기준으로도 FFT가 가장 낮은 train loss와 validation loss를 기록하였다. 이는 전체 파라미터를 업데이트하는 FFT가 학습 데이터와 validation 데이터에 가장 강하게 적응했음을 의미한다.

LoRA 계열에서는 rank가 증가할수록 final train loss와 final valid loss가 점진적으로 감소하였다. LoRA r4의 final valid loss는 1.6443, r8은 1.6407, r16은 1.6383으로 나타났다. 이는 rank 증가가 학습 성능 개선으로 이어졌음을 보여준다.

다만 LoRA r8과 r16의 validation loss 차이는 0.0024 수준으로 매우 작았다. 따라서 LoRA r16이 가장 좋은 validation loss를 기록했지만, 추가 파라미터 수 대비 성능 개선 폭은 제한적이었다.

---

### 9.4 번역 예시 비교

정량 평가 결과를 보완하기 위해 Baseline Transformer, FFT, LoRA r16 모델의 실제 번역 예시를 비교하였다. LoRA 모델은 rank 4, 8, 16 중 가장 높은 성능을 보인 rank 16 결과를 대표 예시로 사용하였다.

| Source | Reference | Baseline Output | FFT Output | LoRA r16 Output |
|---|---|---|---|---|
| 다이제를 깔아둔 원형 틀에 반죽을 부어주세요. | Pour the dough in the round mold with Diget biscuits at the bottom. | Please bring a dough on the original frame of the diameter. | Please call the dough on the original frame that is laid off the dine. | Please bring a dough on the original frame that is placed on the diameter. |
| 괜찮은 것 같은데 이건 최대 몇 명까지 잘 수 있나요? | I think it's okay but how many people can sleep in this? | I think it's fine, so how many people can do it? | I think it's okay, how many people can do it? | I think it's fine, so how many people can do it? |
| 유럽인들과 특히 그 지배자들이 유럽대륙 안팎에 끼친 죄악은 크다. | The crimes committed by Europeans and especially by their rulers on and off continental Europe are great. | In particular, the sin of European people and its rulers in the outside of Europe. | The sinners of Europeans and their domains are particularly large in the area of Europe. | In particular, the sin of European people and its rulers in and outside Europe is largely a big sin. |
| 채팅 액션 방법은 하단의 데몬 아이콘을 클릭하면 가능한 액션들을 볼 수 있습니다. | You can check how to proceed with chatting action by clicking the demon icon below and checking available actions. | The chat action method can be seen if you click the lower-level diamond icon. | The chat action method can be seen as possible action if you click the bottom of the diamond icon. | The chat action method can be seen if you click the H ?? s Done icon. |

예시를 보면 세 모델 모두 문장의 전체적인 구조는 어느 정도 생성하지만, 세부 의미를 정확하게 보존하지 못하는 경우가 있었다. 특히 두 번째 예시에서 “잘 수 있나요?”라는 표현은 “sleep”으로 번역되어야 하지만, 세 모델 모두 “can do it” 형태로 번역하여 핵심 의미를 누락하였다.

FFT는 전체 정량 지표에서는 가장 높은 성능을 보였지만, 개별 예시에서는 항상 가장 자연스러운 번역을 생성하지는 않았다. 예를 들어 첫 번째 문장에서 FFT는 “call the dough”, “laid off the dine”과 같이 의미가 어색한 표현을 생성하였다.

LoRA r16은 Baseline과 비슷한 수준의 문장 구조를 생성하면서 일부 문장에서 약간 더 구체적인 표현을 생성하였다. 그러나 “Diget biscuits”, “sleep”, “demon icon”과 같은 핵심 단어를 정확히 반영하지 못한 사례가 있어, 의미 보존 측면에서는 한계가 있었다.

따라서 정성적 예시 기준으로도 FFT가 전체적으로 가장 높은 성능을 보였다고 단정하기보다는, 평균적인 정량 지표에서는 FFT가 우세하지만 개별 문장에서는 모델별 오류 양상이 다르게 나타난다고 해석할 수 있다.

---

### 9.5 종합 분석

전체 번역 품질은 FFT가 가장 우수하였다. FFT는 Test Loss, PPL, BLEU, chrF, BERTScore 모든 지표에서 가장 좋은 결과를 기록하였다. 또한 학습 로그에서도 가장 낮은 final train loss와 final valid loss를 보였다. 이는 전체 Transformer 파라미터를 모두 fine-tuning하는 방식이 가장 강한 task adaptation 성능을 가진다는 점을 보여준다.

LoRA는 FFT보다 낮은 성능을 보였지만, Baseline Transformer보다는 일관되게 높은 성능을 보였다. LoRA rank 4, 8, 16 모두 Baseline보다 BLEU, chrF, BERTScore F1이 개선되었다.

LoRA rank별 성능은 다음과 같은 순서를 보였다.

- Test Loss: r16 < r8 < r4
- PPL: r16 < r8 < r4
- BLEU: r16 > r8 > r4
- chrF: r16 > r8 > r4
- BERTScore F1: r16 > r8 > r4
- Final Valid Loss: r16 < r8 < r4

따라서 본 실험에서는 rank가 증가할수록 성능이 소폭 개선되는 경향을 확인할 수 있었다. 그러나 rank 증가에 따른 성능 향상 폭은 크지 않았다. r4에서 r16으로 증가했을 때 BLEU는 약 0.1180 상승했고, BERTScore F1은 약 0.0002 상승하였다.

반면 파라미터 효율성 측면에서는 LoRA의 장점이 뚜렷했다. LoRA r8은 전체 파라미터 중 약 0.4842%만 학습했음에도 Baseline보다 높은 번역 성능을 기록하였다. 또한 FFT 대비 약 9.14% 짧은 학습 시간과 약 20% 낮은 peak GPU memory를 기록하였다.

최종적으로 FFT는 최고 성능 모델로 해석할 수 있고, LoRA는 제한된 학습 파라미터와 낮은 메모리 사용량으로 Baseline보다 높은 성능을 얻을 수 있는 효율적인 대안으로 해석할 수 있다. LoRA 계열 중에서는 r16이 가장 높은 성능을 보였지만, 성능과 효율의 균형을 고려하면 r8이 가장 합리적인 설정으로 볼 수 있다.

---

## 10. 분석 및 논의

본 실험에서는 한국어-영어 번역 task에서 Baseline Transformer, FFT, LoRA rank 4/8/16 모델을 비교하였다.

실험 결과, 전체 파라미터를 fine-tuning한 FFT가 가장 높은 번역 품질을 보였다. FFT는 BLEU 30.8565, chrF 58.0224, BERTScore F1 0.9445를 기록하여 모든 모델 중 가장 높은 성능을 보였다. 이는 전체 모델 파라미터를 업데이트하는 방식이 task adaptation 측면에서 가장 강한 성능을 낼 수 있음을 보여준다.

반면 LoRA는 FFT보다 낮은 성능을 보였지만, Baseline Transformer보다는 일관되게 높은 성능을 보였다. 특히 LoRA r16은 LoRA 계열 중 가장 높은 BLEU 27.8453, chrF 55.2860, BERTScore F1 0.9401을 기록하였다.

파라미터 효율성 측면에서는 LoRA의 장점이 뚜렷했다. LoRA r8은 전체 파라미터 중 약 0.4842%만 학습했음에도 Baseline보다 높은 번역 성능을 기록하였다. 또한 FFT보다 학습 시간이 약 9.14% 짧았고, peak GPU memory는 약 20% 낮았다.

LoRA rank 증가에 따른 성능 향상은 확인되었지만, 향상 폭은 제한적이었다. LoRA r4에서 r16으로 갈수록 BLEU, chrF, BERTScore F1, validation loss가 개선되었지만, r8과 r16 사이의 성능 차이는 크지 않았다. 반면 r16은 r8보다 trainable parameter 수가 두 배 많다.

따라서 최고 LoRA 성능만 고려하면 r16이 가장 적절하지만, 성능과 파라미터 효율, 학습 시간, GPU memory를 함께 고려하면 r8이 더 합리적인 선택으로 볼 수 있다.

결론적으로, FFT는 최고 성능을 위한 방식이고, LoRA는 제한된 학습 파라미터와 낮은 자원 사용량으로 효율적인 성능 개선을 얻기 위한 방식으로 해석할 수 있다.

---

## 11. 한계점

이 프로젝트에는 다음과 같은 한계가 있다.

첫째, 실험은 Google Colab Pro의 제한된 GPU 환경에서 수행되었다. 따라서 대규모 기계번역 시스템과 비교했을 때 모델 크기와 dataset 크기에 제약이 있다.

둘째, Baseline Transformer는 학습 및 실험 목적의 직접 구현 모델이다. 따라서 대규모 pretrained translation model과 직접적으로 경쟁하는 것을 목표로 하지 않는다.

셋째, 한국어-영어 번역은 어순, 형태소, 조사, 문장 구조 차이가 크기 때문에 dataset 크기와 모델 규모에 따라 번역 품질에 한계가 있을 수 있다.

---

## 12. 향후 개선 방향

향후 개선할 수 있는 부분은 다음과 같다.

- Beam search decoding 적용
- LoRA target module별 비교 실험
- Attention-only LoRA와 FFN-only LoRA 비교
- 한국어 tokenization 방식 개선
- Checkpoint 기반 중간 평가 추가
- LoRA 외 다른 parameter-efficient fine-tuning 방법과 비교
- Pretrained Transformer 기반 모델로 확장

---

## 13. 프로젝트 파일 구조

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

## 14. 현재 진행 상태

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

## 15. 참고 자료

- Vaswani et al., "Attention Is All You Need"
- The Annotated Transformer
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
