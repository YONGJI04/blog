---
layout: post
title: "[논문 리뷰] Auto-Encoding Variational Bayes (VAE)"
date: 2026-07-24 10:00:00+0900
description: 잠재변수 모델을 신경망으로 근사하고 reparameterization trick으로 학습 가능하게 만든 VAE 논문 정리
tags: vae generative-model variational-inference
categories: Generative AI
related_posts: false
toc:
  sidebar: left
---

- **논문**: Auto-Encoding Variational Bayes
- **저자**: Diederik P. Kingma, Max Welling
- **학회**: ICLR 2014

## 한 줄 요약

계산 불가능한(intractable) 사후분포를 신경망(인코더)으로 근사하고, reparameterization trick으로 샘플링 과정을 미분 가능하게 만들어서 잠재변수 생성모델을 확률적 경사하강법으로 직접 학습할 수 있게 만든 논문. GAN과 함께 딥러닝 생성모델의 양대 축 중 하나가 되었다.

## 배경

데이터 $x$가 관측되지 않은 잠재변수 $z$로부터 생성된다고 가정하는 잠재변수 모델을 생각해보자.

$$
p_\theta(x) = \int p_\theta(x|z)\, p(z)\, dz
$$

이 모델을 학습하려면 사후분포 $p_\theta(z|x)$가 필요한데, 위 적분이 계산 불가능한 경우가 대부분이라 $p_\theta(z|x)$도 closed-form으로 구할 수 없다. 기존 변분추론(Variational Inference)은 $p_\theta(z|x)$를 다루기 쉬운 분포 $q(z)$로 근사하되, 주로 mean-field 가정과 좌표 상승법(coordinate ascent)으로 데이터 포인트마다 별도로 최적화했다. 이 방식은 대규모 데이터셋이나 복잡한 신경망 기반 근사분포에는 계산 비용이 너무 커서 그대로 쓰기 어려웠다.

## 제안 방법

### 인코더-디코더 구조

- **인식 네트워크(encoder)** $q_\phi(z|x)$: 입력 $x$를 받아 잠재분포의 파라미터(평균 $\mu$, 분산 $\sigma^2$)를 출력. 보통 $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x)I)$로 둔다.
- **생성 네트워크(decoder)** $p_\theta(x|z)$: 잠재변수 $z$를 받아 $x$를 복원.

### ELBO (Evidence Lower BOund)

직접 최적화할 수 없는 $\log p_\theta(x)$ 대신, 다음 하한(lower bound)을 최대화한다.

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)
$$

첫 번째 항은 재구성(reconstruction) 항으로, 인코더가 만든 $z$로부터 $x$를 얼마나 잘 복원하는지를 나타낸다. 두 번째 항은 근사 사후분포 $q_\phi(z|x)$가 사전분포 $p(z)$(보통 표준정규분포)에서 얼마나 벗어나는지에 대한 정규화 역할을 한다. $p(z)$와 $q_\phi(z|x)$를 둘 다 가우시안으로 두면 KL항은 closed-form으로 계산할 수 있다.

### Reparameterization Trick

문제는 $\mathbb{E}_{q_\phi(z|x)}[\cdot]$ 항을 역전파하려면 샘플링 $z \sim q_\phi(z|x)$ 과정 자체가 미분 가능해야 한다는 점이다. 확률적 샘플링은 그 자체로 미분이 안 되므로, 다음과 같이 노이즈를 분리한다.

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

이렇게 하면 확률성은 $\epsilon$이라는 외부 노이즈로 빠지고, $\mu_\phi$와 $\sigma_\phi$에 대한 gradient는 정상적으로 역전파된다. 이 트릭 덕분에 인코더와 디코더를 하나의 계산 그래프로 묶어 표준 SGD로 end-to-end 학습이 가능해졌다.

## 실험 결과

MNIST, Frey Face 데이터셋에서 기존 변분추론 방법 대비 더 나은 marginal likelihood 하한을 보였고, 학습된 잠재공간에서 두 샘플 사이를 보간(interpolation)했을 때 자연스럽게 변화하는 생성 결과를 통해 잠재공간이 의미 있는 구조로 정리된다는 것을 시각적으로 보여줬다.

## 느낀 점

- ELBO를 신경망 파라미터에 대해 직접 미분 가능하게 만든 것(reparameterization trick)이 이 논문의 핵심이고, 이후 등장한 β-VAE, VQ-VAE, 그리고 확산모델(diffusion model)의 변분적 해석까지 이어지는 이론적 뼈대가 됐다.
- 가우시안 디코더를 쓸 때 생성 이미지가 다소 흐릿(blurry)해지는 경향은 VAE의 잘 알려진 한계인데, 이는 pixel-wise 재구성 손실이 불확실성을 평균으로 흡수해버리기 때문이다. 이후 VQ-VAE처럼 discrete latent를 쓰거나 GAN/디퓨전과 결합하는 방향으로 많이 보완됐다.
- GAN과 비교하면 학습이 안정적이고 명시적인 likelihood(하한)를 최적화한다는 장점이 뚜렷해서, 왜 이 논문이 여전히 생성모델 입문 논문으로 꼽히는지 이해가 됐다.
