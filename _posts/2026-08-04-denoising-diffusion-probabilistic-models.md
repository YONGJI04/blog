---
layout: post
title: "[논문 리뷰] Denoising Diffusion Probabilistic Models (DDPM)"
date: 2026-08-04 09:00:00+0900
description: 데이터에 점진적으로 노이즈를 주입하는 forward process와 그걸 거꾸로 복원하도록 학습하는 reverse process로 생성모델을 구성한 DDPM 논문 정리
tags: diffusion ddpm generative-model score-matching
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Denoising Diffusion Probabilistic Models
- **저자**: Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
- **학회**: NeurIPS 2020

## 한 줄 요약

데이터에 $T$ 스텝에 걸쳐 조금씩 가우시안 노이즈를 섞어 완전한 노이즈로 만드는 **forward process**를 고정해두고, 신경망이 그 반대 방향(노이즈 → 데이터)으로 한 스텝씩 복원하는 **reverse process**를 학습하게 만드는 생성모델. 학습 목적함수가 결국 "각 스텝에 섞인 노이즈를 예측하는" 단순한 회귀 손실로 정리되면서, [GAN](/blog/2026/generative-adversarial-networks/)의 불안정한 적대적 학습이나 [VAE](/blog/2026/auto-encoding-variational-bayes/)의 흐릿한 샘플 문제 없이 고품질 샘플을 만들어냈고, 이후 Stable Diffusion, DALL-E 2, Imagen 등 현재 이미지 생성모델 대부분의 근간이 되었다.

## 배경

이 논문 이전까지 딥러닝 생성모델은 대체로 두 갈래였다.

- **GAN**: 적대적 학습으로 선명한 샘플을 만들지만, 학습이 불안정하고 mode collapse에 취약하며 명시적 우도가 없다.
- **VAE**: ELBO를 직접 최적화해서 학습은 안정적이지만, 사후분포를 단순한 형태(주로 가우시안)로 근사하다 보니 샘플이 흐릿해지는 경향이 있다.

한편 diffusion 모델 자체는 Sohl-Dickstein et al. (2015)이 비평형 열역학(non-equilibrium thermodynamics)에서 아이디어를 가져와 먼저 제안했지만, 그때는 샘플 품질이 GAN에 크게 못 미쳤다. DDPM은 이 프레임워크를 다시 가져와서, **목적함수를 노이즈 예측이라는 형태로 재정식화**하면 학습이 훨씬 쉬워지고 실제로 GAN급 샘플 품질이 나온다는 걸 보였다.

## 제안 방법

### Forward Process (노이즈 주입)

데이터 $x_0$에서 시작해서, 정해진 분산 스케줄 $\beta_1, \dots, \beta_T$에 따라 매 스텝 조금씩 가우시안 노이즈를 더한다. 이 과정은 학습 파라미터가 없는 고정된 마르코프 체인이다.

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I\right)
$$

중요한 성질은, 이 체인을 여러 스텝 합성해도 여전히 닫힌 형태의 가우시안이라는 것이다. $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$라 하면

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\right)
$$

즉 임의의 스텝 $t$의 노이즈 낀 이미지를 매 스텝 반복하지 않고 **한 번에** 샘플링할 수 있다: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. $T$가 충분히 크면 $x_T$는 거의 순수한 표준 가우시안 노이즈가 된다.

### Reverse Process (노이즈 제거, 학습 대상)

생성은 반대 방향이다. 순수 노이즈 $x_T \sim \mathcal{N}(0,I)$에서 시작해서, 신경망 $p_\theta$가 한 스텝씩 노이즈를 걷어내며 $x_0$까지 복원한다.

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)
$$

$q(x_{t-1}\mid x_t, x_0)$ 자체는 $x_0$를 알면 닫힌 형태의 가우시안이라는 사실(베이즈 정리로 유도)을 이용해서 variational lower bound를 전개하면, 각 스텝의 손실이 결국 $q$와 $p_\theta$ 두 가우시안 사이의 KL divergence, 즉 **평균들 간의 거리**로 정리된다.

### 핵심 재정식화: 노이즈 예측으로 단순화

논문의 핵심 기여는 여기서다. $\mu_\theta(x_t,t)$를 직접 예측하는 대신, forward process 식 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$을 이용해 $\mu_\theta$를 노이즈 $\epsilon$에 대한 식으로 재매개변수화(reparameterize)하면, 학습 목적함수가 극도로 단순해진다.

$$
L_{simple}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]
$$

즉 신경망 $\epsilon_\theta$는 **"이 시점 $t$에 이 이미지 $x_t$에 실제로 섞여 있는 노이즈 $\epsilon$이 뭐였는지"만 예측**하도록 MSE로 학습한다. VAE의 ELBO 항이나 GAN의 min-max 게임에 비해 압도적으로 단순한, 이미지 디노이징(denoising) 회귀 문제가 된 것이다.

### 샘플링

학습된 $\epsilon_\theta$로 $x_T$에서 $x_0$까지 $T$ 스텝을 거꾸로 반복한다.

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0,I)
$$

매 스텝 신경망을 한 번씩 호출해야 하므로, $T$(논문에서는 1000)만큼 forward pass를 반복해야 샘플 하나가 나온다 — 이게 뒤에서 언급할 가장 큰 실용적 단점이다.

### 아키텍처

$\epsilon_\theta(x_t, t)$는 U-Net 구조를 사용한다. 인코더-디코더 사이에 skip connection이 있어 저수준 디테일을 보존하고, 시간 스텝 $t$는 Transformer의 positional encoding과 같은 방식(sinusoidal embedding)으로 인코딩해서 각 레이어에 주입한다.

## 실험 결과

- CIFAR-10에서 FID(Fréchet Inception Distance) 3.17을 달성해, 당시 대부분의 GAN 기반 모델을 능가.
- LSUN 데이터셋(교회, 침실 등)에서도 고해상도의 사실적인 이미지 생성.
- Inception Score 등 정량 지표에서 우도 기반 모델(NLL 최적화) 대비 뚜렷한 샘플 품질 우위를 보임 — 저자들도 지적하듯, DDPM의 손실은 lossless codelength 관점에서는 최적이 아니지만 지각적(perceptual) 품질에서는 오히려 더 낫다.

## 느낀 점

- GAN 리뷰에서 "판별자를 속인다"는 프레이밍, VAE 리뷰에서 "잠재변수의 사후분포를 근사한다"는 프레이밍을 봤는데, DDPM은 문제를 아예 **"이미지에 낀 노이즈를 맞히는 회귀"**로 바꿔버린 게 제일 인상적이다. 목적함수가 단순해질수록(MSE) 학습이 안정적이라는 걸 세 논문을 나란히 보면서 체감했다.
- 다만 이 안정성의 대가가 명확하다 — 샘플 하나 만드는 데 $T=1000$번의 신경망 forward pass가 필요해서, GAN(1회 forward)이나 VAE(1회 forward)에 비해 추론이 압도적으로 느리다. 이후 DDIM, latent diffusion(Stable Diffusion) 같은 후속 연구들이 정확히 이 "샘플링 스텝 수" 문제를 줄이는 데 집중한 이유가 여기서 바로 이해가 됐다.
- $L_{simple}$이 사실은 원래의 variational lower bound에서 각 항의 가중치를 다 무시하고 단순 MSE로 근사한 것인데, 이 "이론적으로는 느슨한 근사"가 오히려 실제 샘플 품질을 더 좋게 만든다는 점이 흥미롭다 — 우도 최적화와 지각 품질이 항상 같은 방향이 아니라는 걸 보여주는 사례로 느껴졌다.
