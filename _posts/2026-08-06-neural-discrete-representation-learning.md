---
layout: post
title: "[논문 리뷰] Neural Discrete Representation Learning (VQ-VAE)"
date: 2026-08-06 09:00:00+0900
description: 연속적인 잠재변수를 벡터 양자화로 이산 코드로 바꾸어 의미 있는 표현과 강력한 생성모델을 만드는 VQ-VAE 논문 정리
tags: vq-vae vae generative-model representation-learning
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Neural Discrete Representation Learning
- **저자**: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (DeepMind)
- **학회**: NeurIPS 2017

## 한 줄 요약

인코더가 만든 연속적인 잠재벡터를 미리 정의한 코드북의 가장 가까운 벡터로 양자화해 **이산(discrete) 잠재 표현**을 학습하는 오토인코더. 연속 잠재공간을 쓰는 VAE의 한계를 피하면서, 이미지·음성·비디오를 더 compact한 토큰 시퀀스로 표현할 수 있게 만든 논문이다.

## 배경

기존 VAE는 보통 $q\_\phi(z \vert x)$를 가우시안으로 두고, 잠재변수 $z$가 표준정규분포를 따르도록 KL divergence를 사용한다. 학습은 안정적이지만 잠재 표현이 지나치게 매끄럽게 정규화되고, 픽셀 단위 재구성 손실 때문에 이미지가 흐릿해지는 문제가 있다.

반면 이미지나 음성에는 연속적인 값만으로 설명하기 어려운 고수준의 구성요소가 있다. 예를 들면 이미지의 특정 질감이나 음성의 음소처럼, 데이터가 반복해서 사용하는 **이산적인 패턴의 목록**을 먼저 학습하면 더 효율적인 표현이 될 수 있다.

## 제안 방법

### 코드북과 벡터 양자화

VQ-VAE는 $K$개의 임베딩 벡터로 이루어진 코드북 $\{e\_1, \dots, e\_K\}$를 학습한다. 인코더 출력 $z\_e(x)$가 주어지면 가장 가까운 코드 벡터를 선택한다.

$$
z_q(x) = e_k, \qquad k = \arg\min_j \|z_e(x) - e_j\|_2
$$

이렇게 하면 디코더는 연속적인 실수 벡터 대신 코드북 인덱스의 격자(grid)를 입력으로 받는다. 각 위치에서 $K$개 코드 중 하나를 고르므로, 잠재 표현 전체는 이미지마다 다른 이산 토큰 시퀀스가 된다.

### 학습 손실

양자화 연산의 $\operatorname{argmin}$은 미분할 수 없기 때문에, 논문은 straight-through estimator를 사용한다. 순전파에서는 양자화된 $z_q$를 사용하고, 역전파에서는 $z_q$를 $z_e$를 통과하는 것처럼 gradient를 복사한다.

전체 손실은 재구성 손실과 코드북 학습 손실, 그리고 인코더 출력이 선택된 코드에 가까워지도록 하는 commitment 손실로 구성된다.

$$
L = \log p(x \vert z_q(x)) + \|\operatorname{sg}[z_e(x)] - e_k\|_2^2 + \beta\|z_e(x) - \operatorname{sg}[e_k]\|_2^2
$$

여기서 $\operatorname{sg}[\cdot]$는 stop-gradient 연산이다. 두 번째 항은 코드북 벡터를 인코더 출력 쪽으로 이동시키고, 세 번째 항은 인코더가 선택한 코드에서 너무 멀리 벗어나지 않도록 붙잡아 둔다.

### Prior와 생성

VQ-VAE의 인코더와 디코더만 학습하면 입력을 재구성하는 모델이 된다. 새로운 이미지를 생성하려면 잠재 코드 인덱스들의 분포를 따로 학습해야 한다. 논문은 잠재 코드에 PixelCNN prior를 학습한 뒤, PixelCNN에서 샘플링한 코드 시퀀스를 디코더에 넣어 이미지를 생성한다.

즉 생성 과정은 다음처럼 분리된다.

- 인코더: 이미지 → 이산 코드
- prior: 코드들의 공간적 의존성 학습 및 코드 시퀀스 샘플링
- 디코더: 이산 코드 → 이미지

이 구조 덕분에 prior는 원본 픽셀 공간이 아니라 훨씬 짧은 잠재 공간에서 동작할 수 있다.

## 실험 결과

MNIST와 CIFAR-10 이미지에서 의미 있는 discrete representation을 학습했고, 얼굴·객체·음성 데이터에서도 코드북이 데이터의 중요한 패턴을 포착하는 것을 보였다. 특히 음성 실험에서는 프레임 단위의 연속적인 신호를 이산 코드로 바꾼 뒤, 코드 prior가 음성의 장거리 구조를 모델링할 수 있음을 확인했다.

## 느낀 점

- VAE가 확률적인 연속 잠재공간을 정리했다면, VQ-VAE는 잠재공간을 **토큰화**해서 이후 Transformer가 다루기 좋은 형태로 바꿨다는 점이 핵심이다.
- 코드북 크기와 잠재 해상도는 표현력과 압축률을 결정한다. 코드가 너무 적으면 정보가 손실되고, 너무 많으면 코드가 제대로 재사용되지 않을 수 있다.
- 이후 VQGAN과 이미지 토큰 기반 Transformer, 그리고 여러 멀티모달 모델로 이어지는 흐름의 출발점이라서, 이미지 생성모델을 픽셀 예측에서 토큰 예측 문제로 바꾼 논문으로 볼 수 있다.
