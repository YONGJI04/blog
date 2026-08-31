---
layout: post
title: "[논문 리뷰] Taming Transformers for High-Resolution Image Synthesis (VQGAN)"
date: 2026-08-08 09:00:00+0900
description: VQGAN으로 이미지를 이산 토큰으로 압축하고 Transformer prior로 고해상도 이미지를 생성하는 논문 정리
tags: vqgan vq-vae transformer image-generation
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Taming Transformers for High-Resolution Image Synthesis
- **저자**: Patrick Esser, Robin Rombach, Björn Ommer (Heidelberg University)
- **학회**: CVPR 2021

## 한 줄 요약

VQ-VAE의 코드북 기반 오토인코더에 **GAN 손실과 perceptual loss**를 결합해 더 선명한 이미지 토큰을 만들고, 그 토큰의 분포를 Transformer로 학습해 고해상도 이미지를 생성하는 방법. 이후 Latent Diffusion과 Stable Diffusion으로 이어지는 latent-space 생성의 중요한 연결고리다.

## 배경

Transformer는 이미지의 픽셀을 직접 autoregressive하게 예측할 수 있지만, 고해상도 이미지에서는 토큰 수가 너무 많아진다. 예를 들어 $256 \times 256$ 이미지를 픽셀 단위로 다루면 한 장이 65,536개의 토큰이 되므로 self-attention과 순차 생성 비용이 매우 커진다.

VQ-VAE는 이미지를 더 작은 이산 코드 시퀀스로 압축해 이 문제를 줄였지만, 단순한 픽셀 재구성 손실만 사용하면 고주파 디테일이 평균화되어 결과가 흐릿해질 수 있다. VQGAN은 이 압축 모델을 GAN으로 학습해서, 짧으면서도 시각적으로 중요한 정보를 보존하는 잠재 토큰을 만들고자 한다.

## 제안 방법

### VQGAN autoencoder

VQGAN은 인코더 $E$, 코드북 $\{z\_k\}$, 디코더 $G$로 이미지를 잠재 토큰으로 바꾼다.

$$
z_q(x) = \arg\min_{z_k} \|E(x) - z_k\|_2
$$

디코더는 양자화된 잠재 표현으로 이미지를 복원한다. VQ-VAE와의 중요한 차이는 복원 품질을 단순한 pixel-wise loss만으로 평가하지 않는다는 점이다.

### Perceptual loss와 adversarial loss

먼저 사전학습된 네트워크의 feature 공간에서 원본과 복원 이미지의 차이를 계산한다. 이 perceptual loss는 픽셀의 작은 위치 차이보다 사람이 인지하는 구조와 질감의 유사성에 더 민감하다.

여기에 판별자 $D$를 추가해 복원 이미지가 실제 이미지처럼 보이도록 adversarial loss를 학습한다.

$$
L_{GAN} = \log D(x) + \log(1 - D(\hat{x})), \qquad \hat{x} = G(z_q(x))
$$

전체적으로는 재구성 손실, codebook 손실, perceptual loss, adversarial loss를 균형 있게 조합한다. 판별자가 모든 픽셀을 외우지 않도록 patch 단위 판별자를 사용해 국소적인 질감과 디테일을 평가한다.

### Transformer prior

VQGAN의 인코더는 이미지를 $f \times f$만큼 줄인 $h \times w$개의 잠재 토큰으로 바꾼다. Transformer는 이 토큰을 일렬로 펼친 뒤 다음 토큰을 예측하는 autoregressive prior를 학습한다.

$$
p(s) = \prod_{i=1}^{hw} p(s_i \vert s_{<i})
$$

학습이 끝나면 Transformer에서 토큰 시퀀스를 샘플링하고, 이를 VQGAN 디코더에 넣어 이미지를 복원한다. 고해상도에서는 전체 이미지를 한 번에 처리하지 않고, 이미지를 여러 영역으로 나눠 coarse-to-fine 방식으로 생성하는 sliding-window attention도 사용한다.

## VQ-VAE와의 차이

- **VQ-VAE**: 코드북과 재구성 중심. 압축된 표현을 안정적으로 학습하는 데 초점.
- **VQGAN**: perceptual loss와 GAN을 추가. 짧은 코드에서도 시각적으로 중요한 디테일을 보존하는 데 초점.
- **생성 prior**: 둘 다 잠재 코드 위에 별도의 prior를 학습할 수 있지만, VQGAN은 고해상도 이미지 생성을 위해 Transformer 구조와 영역별 attention을 적극적으로 활용한다.

## 실험 결과

ImageNet과 풍경·건축물 등 다양한 데이터셋에서, 같은 잠재 해상도에서도 VQGAN이 기존 VQ-VAE보다 더 높은 재구성 품질과 선명한 디테일을 보였다. 또한 이미지 전체를 픽셀 단위로 예측하는 Transformer보다 훨씬 짧은 잠재 시퀀스에서 autoregressive generation을 수행하면서 고해상도 이미지 합성이 가능함을 보였다.

## 느낀 점

- VQGAN의 핵심은 GAN을 최종 이미지 생성기에만 쓰지 않고, **Transformer가 예측할 잠재 토큰 자체를 더 좋은 시각적 단위로 만드는 데 사용했다**는 점이다.
- VQ-VAE에서 시작한 “이미지를 토큰으로 바꾼다”는 발상이 VQGAN에서는 “좋은 토큰을 만들고 토큰의 언어모델을 학습한다”는 구조로 확장된다. 이미지 생성과 언어모델링 사이의 연결이 선명하게 보이는 지점이다.
- GAN을 넣은 만큼 코드북 붕괴나 학습 불안정성 같은 새로운 문제가 생긴다. 이후 latent diffusion 계열은 이 아이디어를 계승하면서 autoregressive sampling의 느린 속도와 GAN 학습의 불안정성을 함께 줄이는 방향으로 발전했다.
