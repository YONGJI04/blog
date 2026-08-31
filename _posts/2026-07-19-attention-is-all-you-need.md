---
layout: post
title: "[논문 리뷰] Attention Is All You Need (Transformer)"
date: 2026-07-19 09:00:00+0900
description: RNN과 CNN 없이 어텐션만으로 시퀀스 변환 모델을 구성한 Transformer 논문 정리
tags: transformer attention seq2seq
categories: NLP
related_posts: false
toc:
  sidebar: left
---

- **논문**: Attention Is All You Need
- **저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar 외 (Google Brain / Google Research)
- **학회**: NeurIPS 2017

## 한 줄 요약

RNN이나 CNN 없이 어텐션(attention) 메커니즘만으로 인코더-디코더 시퀀스 변환 모델을 구성해서, 학습 병렬화와 장거리 의존성(long-range dependency) 학습을 동시에 해결한 논문. 이후 BERT, GPT 계열을 포함한 거의 모든 LLM의 기반 아키텍처가 되었다.

## 배경

기존 seq2seq 모델은 대부분 RNN(LSTM/GRU) 기반이었다. RNN은 토큰을 순서대로 하나씩 처리하기 때문에 다음과 같은 한계가 있었다.

- **순차 계산**: 시점 $t$의 은닉 상태가 $t-1$에 의존하므로 시퀀스 내에서 병렬화가 불가능하다. 시퀀스가 길어질수록 학습 시간이 선형으로 늘어난다.
- **장거리 의존성**: 멀리 떨어진 두 토큰이 서로 영향을 주고받으려면 그 사이의 모든 시점을 거쳐야 해서, 정보가 소실되기 쉽다.

CNN 기반 모델(ConvS2S 등)은 병렬화는 가능했지만, 커널 크기가 고정되어 있어 먼 위치 간의 관계를 포착하려면 레이어를 깊게 쌓아야 했다. Transformer는 아예 recurrence와 convolution을 모두 걷어내고 attention만으로 이 문제를 풀었다.

## 제안 방법

### Scaled Dot-Product Attention

쿼리 $Q$, 키 $K$, 값 $V$ 행렬에 대해 다음과 같이 계산한다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

내적 $QK^T$를 $\sqrt{d_k}$로 스케일링하는 이유는, $d_k$가 커질수록 내적 값의 분산이 커져 softmax가 극단적인(거의 one-hot에 가까운) 분포로 saturate되고, 이로 인해 gradient가 사라지는 문제를 완화하기 위해서다.

### Multi-Head Attention

$Q$, $K$, $V$를 서로 다른 학습된 projection으로 $h$번 나눠서 각각 attention을 병렬로 계산한 뒤 concat한다.

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

헤드마다 서로 다른 표현 부분공간(subspace)에 주목할 수 있게 해서, 단일 attention보다 다양한 관계(구문적, 의미적 등)를 동시에 학습할 수 있다.

### Positional Encoding

Recurrence를 없앴기 때문에 모델은 토큰의 순서 정보를 알 방법이 없다. 이를 sin/cos 함수 기반의 위치 인코딩을 입력 임베딩에 더해서 주입한다.

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 전체 구조

- 인코더/디코더 각각 동일한 구조의 레이어를 $N$번 쌓음 (논문에서는 $N=6$)
- 각 레이어는 Self-Attention → residual connection + LayerNorm → Feed-Forward → residual connection + LayerNorm 순서
- 디코더는 미래 토큰을 못 보게 마스킹한 self-attention(masked self-attention)을 사용해 auto-regressive하게 생성
- 디코더에는 인코더 출력을 참조하는 encoder-decoder attention이 추가로 존재

## 실험 결과

WMT 2014 English-to-German, English-to-French 번역 태스크에서 당시 최고 성능(SOTA) BLEU 점수를 달성했다. 특히 학습에 필요한 연산량(FLOPs)이 기존 SOTA 모델 대비 훨씬 적었고, RNN과 달리 시퀀스 내부를 병렬로 처리할 수 있어 학습 시간도 크게 단축되었다.

## 느낀 점

- 구조 자체는 지금 기준으로 보면 단순한데, "recurrence를 완전히 없앤다"는 선택이 결국 GPU 병렬 학습 시대와 맞아떨어져서 이후 스케일링의 문을 열었다는 점이 인상적이다.
- Self-attention의 계산/메모리 복잡도가 시퀀스 길이 $n$에 대해 $O(n^2)$라는 점은 이 논문의 명확한 한계이고, 이후 Longformer, Linformer, FlashAttention 등 효율적 attention 연구들이 이 지점을 계속 파고들고 있다.
- Positional encoding을 고정된 sin/cos 함수로 준 것도 이후 RoPE, ALiBi 같은 상대적 위치 인코딩 방식으로 계속 개선되어 온 부분이라, "최초 버전"으로서의 의미를 다시 느꼈다.
