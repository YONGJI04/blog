---
layout: post
title: "[논문 리뷰] Generative Adversarial Networks (GAN)"
date: 2026-08-04 09:00:00+0900
description: 생성자와 판별자를 적대적으로 경쟁시켜 명시적 우도 계산 없이 데이터 분포를 학습하는 GAN 논문 정리
tags: gan generative-model adversarial-training
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Generative Adversarial Networks
- **저자**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (Université de Montréal)
- **학회**: NeurIPS (NIPS) 2014

## 한 줄 요약

생성자(Generator)와 판별자(Discriminator) 두 신경망을 min-max 게임으로 서로 경쟁시켜서, 명시적으로 우도(likelihood)를 계산하거나 근사 추론을 하지 않고도 데이터 분포를 흉내 내는 샘플을 만들어내는 프레임워크. 이후 DCGAN, WGAN, CycleGAN, StyleGAN 등 거대한 생성모델 계열의 출발점이 되었다.

## 배경

이 논문 이전의 딥러닝 기반 생성모델은 대부분 **명시적으로 확률분포를 모델링**하려 했다.

- Deep Boltzmann Machine 같은 모델은 분배함수(partition function)가 intractable해서 MCMC 기반 근사가 필요했고, 학습이 느리고 불안정했다.
- [VAE](/blog/2026/auto-encoding-variational-bayes/)는 이 문제를 variational lower bound(ELBO)로 우회했지만, 여전히 사후분포를 근사하는 encoder가 필요하고, ELBO를 직접 최적화하다 보니 생성 샘플이 흐릿(blurry)해지는 경향이 있었다.
- 두 접근 모두 "얼마나 그럴듯한 데이터를 생성했는가"를 **명시적 확률(우도)로 채점**하려는 시도였고, 그 채점 자체가 계산적으로 비싸거나 근사 오차를 동반했다.

GAN은 이 채점 문제 자체를 없애버린다. 데이터가 그럴듯한지 아닌지를 우도 대신 **또 다른 신경망(판별자)이 직접 분류**하게 만들고, 생성자는 그 판별자를 속이는 방향으로만 학습한다. 즉 "얼마나 진짜 같은가"를 확률식으로 풀지 않고 이진 분류 문제로 치환한 것이다.

## 제안 방법

### Minimax 게임

생성자 $G$는 노이즈 $z \sim p_z(z)$를 받아 데이터 공간의 샘플 $G(z)$를 만든다. 판별자 $D$는 입력이 실제 데이터인지 $G$가 만든 가짜인지 확률을 출력한다. 둘은 다음 값함수(value function)에 대해 반대 목표로 경쟁한다.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

- $D$는 진짜 $x$에는 $D(x) \to 1$, 가짜 $G(z)$에는 $D(G(z)) \to 0$이 되도록 이 값을 **최대화**한다 (제대로 구분하는 이진 분류기).
- $G$는 $D$가 가짜를 진짜로 착각하도록, 즉 $D(G(z)) \to 1$이 되도록 이 값을 **최소화**한다.

두 네트워크는 서로의 손실에 직접 개입하는 상대이므로, 지도학습처럼 고정된 타겟에 수렴하는 게 아니라 **두 플레이어가 동시에 움직이는 게임의 내시 균형(Nash equilibrium)**을 찾는 문제가 된다.

### 이론적 최적점: Jensen-Shannon Divergence

$G$를 고정했을 때 $V(D,G)$를 최대화하는 최적 판별자는 닫힌 형태로 구해진다.

$$
D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

이 $D^*_G$를 다시 대입하면, $G$의 목표는 결국 $p_{data}$와 생성분포 $p_g$ 사이의 **Jensen-Shannon divergence**를 최소화하는 것과 동치임을 논문이 증명한다.

$$
C(G) = -\log 4 + 2 \cdot JSD(p_{data} \,\|\, p_g)
$$

$JSD \geq 0$이고 $p_g = p_{data}$일 때만 $0$이 되므로, 전역 최적해는 정확히 **생성분포가 실제 데이터 분포와 일치하는 지점**($p_g = p_{data}$)이며, 이때 $D^*(x) = \tfrac{1}{2}$ (판별자가 진짜/가짜를 구분 못 하는 상태)로 수렴한다.

### 학습이 실제로는 잘 안 되는 이유와 트릭

이론상으로는 $G$가 $\log(1-D(G(z)))$를 최소화하면 되지만, 학습 초반에는 $G$가 형편없어서 $D$가 가짜를 아주 쉽게 구분한다($D(G(z)) \approx 0$). 이 구간에서 $\log(1-D(G(z)))$는 기울기(gradient)가 거의 0으로 saturate되어 $G$가 학습 신호를 거의 못 받는다.

논문은 이를 해결하기 위해 $G$의 목적함수를 **동일한 고정점을 갖지만 기울기가 훨씬 강한 형태**로 바꿔서 사용한다.

$$
\max_G \ \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]
$$

즉 "$D$가 틀리게 만들 확률을 낮춘다"가 아니라 "$D$가 진짜라고 판단할 확률을 높인다"로 바꾸는 것인데, 수학적 목표는 같지만 초반 학습 구간에서 훨씬 큰 gradient를 준다.

### 학습 절차

$D$와 $G$를 번갈아 업데이트한다.

1. $D$를 $k$ 스텝 동안 SGD로 업데이트 (실제 데이터 배치와 $G$가 만든 가짜 배치로 이진 분류 학습)
2. $G$를 1 스텝 업데이트 ($D$를 속이는 방향)
3. 반복

논문은 이론적으로는 $D$를 매 스텝 최적점까지 학습시켜야 안정적이라고 말하지만, 실험에서는 계산 비용과 오버피팅을 고려해 $k=1$을 사용했다.

## 실험 결과

- MNIST, TFD(Toronto Face Database), CIFAR-10에서 생성 샘플을 정성적으로 제시.
- 정량 평가는 Parzen window 기반 로그우도 추정치를 사용 (GAN 자체가 명시적 우도를 안 주기 때문에 간접적으로 추정할 수밖에 없었음).
- Generator/Discriminator 모두 단순한 MLP로 구성 (지금 기준으로는 아주 작은 모델) — CNN 기반 구조는 이후 DCGAN(2015)에서 도입된다.

## 느낀 점

- "판별자가 못 속이면 생성자가 이긴 것"이라는 프레이밍이 단순하지만, 명시적 밀도 추정이나 근사 추론 없이 샘플링만으로 학습이 된다는 게 이 논문의 진짜 임팩트다. VAE가 ELBO라는 확실한 하한을 최적화하는 것과 대비된다.
- 반면 이 논문에서 이미 나중 GAN 연구 전체를 관통하는 문제들의 씨앗이 보인다. 이론은 $G, D$가 무한한 capacity를 가질 때만 JSD 수렴이 보장되는데, 실제로는 유한한 신경망 파라미터화에서 이 논-convex 게임이 진동하거나 mode collapse(생성자가 데이터 분포의 일부 mode만 만드는 현상)에 빠지기 쉽다.
- $JSD$가 두 분포의 support가 겹치지 않을 때(고차원 데이터에서 흔함) 사실상 상수라 gradient가 사라진다는 점은, 이후 Wasserstein distance를 쓰는 WGAN 계열이 정확히 파고든 지점이다.
- VAE 리뷰에서 "흐릿한 샘플"을 한계로 적었는데, GAN은 반대로 선명한 샘플은 잘 뽑지만 학습 안정성과 분포 커버리지(diversity)를 잃는 쪽으로 트레이드오프가 이동한 셈 — 두 논문을 나란히 보니 생성모델 역사가 이 트레이드오프를 왔다갔다하며 발전해온 게 보인다.
