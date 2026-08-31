---
layout: post
title: "[논문 리뷰] High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)"
date: 2026-08-10 09:00:00+0900
description: 픽셀 공간 대신 압축된 잠재공간에서 diffusion을 수행해 고해상도 이미지 생성을 효율적으로 만든 Stable Diffusion 계열의 기반 논문 정리
tags: stable-diffusion latent-diffusion diffusion text-to-image
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: High-Resolution Image Synthesis with Latent Diffusion Models
- **저자**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
- **학회**: CVPR 2022

## 한 줄 요약

이미지 전체 픽셀 공간에서 diffusion을 수행하는 대신, 먼저 오토인코더로 이미지를 작은 잠재공간으로 압축한 뒤 그 공간에서 노이즈를 제거한다. 여기에 cross-attention을 넣어 텍스트·박스·마스크 같은 조건을 연결함으로써, 품질을 유지하면서 계산량을 크게 줄인 **Latent Diffusion Model(LDM)**을 제안한 논문이다.

## 배경

[DDPM](/blog/2026/denoising-diffusion-probabilistic-models/)은 매우 좋은 이미지 샘플을 만들지만, 픽셀 공간에서 $T$번의 denoising을 수행해야 하므로 학습과 추론 비용이 크다. 고해상도 이미지에서는 각 스텝마다 처리해야 하는 픽셀 수 자체가 너무 많아진다.

반대로 [VQ-GAN](/blog/2026/taming-transformers-for-high-resolution-image-synthesis/)처럼 이미지를 압축된 표현으로 바꾸면 연산량은 줄일 수 있지만, 코드북 기반의 autoregressive Transformer는 토큰을 순차적으로 생성해야 한다. LDM은 이 압축된 잠재공간에서 diffusion을 수행해 두 장점을 결합한다.

## 제안 방법

### Perceptual compression

먼저 이미지 $x$를 인코더 $E$로 잠재 표현 $z = E(x)$로 바꾸고, 디코더 $D$로 복원한다.

$$
z = E(x), \qquad \hat{x} = D(z)
$$

잠재공간의 크기는 원본 이미지보다 훨씬 작지만, 오토인코더가 사람이 인지하기에 중요한 구조와 질감을 보존하도록 학습한다. 논문은 perceptual loss와 patch-based adversarial loss를 사용해 단순한 픽셀 평균화로 인한 흐릿함을 줄인다. 이 점은 [VQ-GAN](/blog/2026/taming-transformers-for-high-resolution-image-synthesis/)의 압축 아이디어와 맞닿아 있지만, LDM은 연속적인 잠재공간에서 diffusion을 수행한다는 차이가 있다.

### Latent diffusion

일반 diffusion이 픽셀 $x_t$에 노이즈를 넣고 제거한다면, LDM은 오토인코더가 만든 잠재 표현 $z$에 같은 과정을 적용한다.

$$
z_t = \sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}\epsilon, \qquad \epsilon \sim \mathcal{N}(0,I)
$$

U-Net $\epsilon_\theta$는 노이즈가 섞인 잠재 표현 $z_t$와 시간 스텝 $t$를 받아 원래 노이즈를 예측한다.

$$
L_{LDM} = \mathbb{E}_{z,c,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t,t,c)\|_2^2\right]
$$

여기서 $c$는 텍스트나 클래스 같은 조건이다. 샘플링이 끝나면 denoised latent $z_0$를 디코더 $D$에 넣어 최종 이미지를 얻는다.

### Cross-attention으로 조건 주입

텍스트 조건을 이미지 잠재공간에 연결하기 위해 U-Net 내부에 cross-attention을 넣는다. 텍스트 인코더가 만든 조건 표현 $\tau_\theta(c)$를 key와 value로 사용하고, 이미지 잠재 feature를 query로 사용한다.

$$
Q = W_Q^{(i)}\varphi_i(z_t), \qquad K = W_K^{(i)}\tau_\theta(c), \qquad V = W_V^{(i)}\tau_\theta(c)
$$

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

이 구조는 텍스트의 각 단어가 이미지의 어느 영역과 관련되는지 학습할 수 있게 한다. 같은 구조를 텍스트뿐 아니라 semantic map, bounding box, 저해상도 이미지 등 다양한 conditioning 입력에도 적용할 수 있다.

### 두 단계 학습

LDM은 크게 두 단계로 학습한다.

1. 오토인코더를 학습해 이미지의 지각적으로 중요한 정보를 잠재공간에 보존한다.
2. 잠재 표현에 noise를 추가하고 제거하는 diffusion U-Net을 학습한다.

이렇게 하면 diffusion 모델은 픽셀의 모든 고주파 세부사항을 직접 처리하지 않고, 오토인코더가 만든 의미 있는 잠재 표현의 분포를 학습하게 된다.

## Stable Diffusion과의 관계

논문의 일반적인 방법론은 Latent Diffusion Model(LDM)이고, Stable Diffusion은 이 아이디어를 오픈 모델과 학습 데이터·조건 설정에 적용한 대표적인 모델 계열이다. 따라서 Stable Diffusion을 이해하려면 DDPM의 diffusion 과정, VAE 계열의 잠재 표현, 그리고 cross-attention 기반 텍스트 조건을 함께 이해해야 한다.

## 실험 결과

ImageNet에서 class-conditional image synthesis를 평가하고, text-to-image·inpainting·semantic image synthesis·super-resolution 등 여러 조건부 생성 태스크를 실험했다. 픽셀 공간 diffusion보다 계산 요구량을 크게 줄이면서도 경쟁력 있는 샘플 품질을 달성했고, 고해상도 이미지에서도 convolutional한 방식으로 생성할 수 있음을 보였다.

## 장점과 한계

- **장점**: 픽셀 공간보다 훨씬 적은 연산으로 고품질 이미지를 만들고, cross-attention을 통해 다양한 조건을 유연하게 받을 수 있다.
- **한계**: 오토인코더가 버린 정보는 diffusion 모델이 복원할 수 없다. 잠재공간의 압축률을 높이면 계산은 줄지만 세부 디테일과 정확한 텍스트 렌더링이 손상될 수 있다.
- **추론 비용**: 픽셀 공간 diffusion보다 효율적이지만 여러 denoising step을 순차적으로 실행해야 하므로 GAN이나 일반적인 한 번의 디코더 실행보다 느리다.

## 느낀 점

- LDM의 핵심은 새로운 diffusion 수식보다 **어디에서 diffusion을 할 것인가**를 바꾼 데 있다. 고차원 픽셀 공간의 불필요한 세부사항을 오토인코더에 맡기고, diffusion은 의미와 구조가 잘 정리된 잠재공간을 담당하게 했다.
- 앞서 본 VQ-VAE와 VQ-GAN이 이미지를 토큰 또는 잠재 표현으로 바꾸는 흐름을 만들었다면, LDM은 그 잠재공간 위에서 안정적인 diffusion 생성까지 연결한다.
- Stable Diffusion이 널리 쓰일 수 있었던 이유도 품질만이 아니라, 이 잠재공간 설계 덕분에 학습과 추론 비용을 현실적인 수준으로 낮췄기 때문이라고 볼 수 있다.
