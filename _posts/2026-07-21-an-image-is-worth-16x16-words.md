---
layout: post
title: "[논문 리뷰] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)"
date: 2026-07-21 09:00:00+0900
description: 이미지를 16x16 패치 시퀀스로 잘라 순수 Transformer 인코더로 분류하는 ViT 논문 정리
tags: vit transformer image-classification pretraining
categories: ["Computer Vision"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **저자**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Research, Brain Team)
- **학회**: ICLR 2021

## 한 줄 요약

이미지를 고정 크기 패치(예: 16x16)로 잘라 각 패치를 "단어"처럼 취급하고, 컨볼루션 없이 **표준 Transformer 인코더**만으로 이미지를 분류한다. 대규모 데이터로 사전학습하면 당시 최고 수준의 CNN을 넘어서면서 학습 비용은 더 적게 들었고, 이후 비전 분야의 백본이 CNN에서 Transformer로 넘어가는 출발점이 되었다.

## 배경

[Transformer](/blog/2026/attention-is-all-you-need/)는 NLP에서 사실상 표준이 되었지만, 컴퓨터 비전에서는 여전히 CNN이 지배적이었다. 기존 시도들은 CNN에 self-attention을 끼워 넣거나 CNN 특징맵 위에 attention을 얹는 식이라, 구조가 복잡하거나 하드웨어에서 효율적으로 확장하기 어려웠다.

이 논문의 질문은 단순하다: **이미지에 Transformer를 최소한의 수정만으로 그대로 적용하면 어떻게 될까?** 이미지 픽셀 하나하나에 attention을 적용하면 시퀀스 길이가 픽셀 수의 제곱으로 늘어 비현실적이므로, 이미지를 패치 단위로 묶어 시퀀스 길이를 줄이는 것이 핵심 아이디어다.

## 제안 방법

### 패치 임베딩

$H \times W \times C$ 이미지를 $P \times P$ 크기 패치 $N = HW/P^2$개로 나누고, 각 패치를 펼쳐(flatten) 하나의 선형 투영 $E$로 $D$차원 벡터로 만든다. NLP 토큰 임베딩에 해당하는 부분이다. 여기에 BERT의 `[CLS]` 토큰처럼 학습 가능한 **class 토큰**을 맨 앞에 붙이고, 패치 순서 정보를 위해 **학습 가능한 위치 임베딩**을 더한다.

$$
z_0 = [x_{class};\, x_p^1 E;\, x_p^2 E;\, \dots;\, x_p^N E] + E_{pos}
$$

### Transformer 인코더

이 시퀀스를 Multi-head Self-Attention(MSA)과 MLP 블록이 번갈아 쌓인 표준 Transformer 인코더에 넣는다. 각 블록은 앞에 LayerNorm을 두고 잔차 연결(residual connection)로 감싼다.

$$
z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad
z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell
$$

마지막 층에서 class 토큰의 출력을 분류 head(MLP)에 넣어 결과를 얻는다. 이미지에 특화된 구조는 패치를 자르는 부분과 위치 임베딩뿐이고, 나머지는 NLP에서 쓰는 Transformer 그대로다.

### 귀납적 편향(inductive bias)의 부재

CNN은 지역성(locality)과 이동 등변성(translation equivariance)이 구조에 내장되어 있다. ViT는 이런 가정이 거의 없고, MLP 층만 국소적·등변적이며 self-attention은 전역적이다. 위치 임베딩도 처음에는 2D 구조를 모르는 상태로 시작해서 학습으로 익혀야 한다. 이것이 아래 실험 결과에서 데이터 규모가 중요한 이유다.

### 모델 크기와 고해상도 파인튜닝

BERT를 따라 Base(12층, 약 86M), Large(24층, 약 307M), Huge(32층, 약 632M) 세 가지 크기를 만들고, `ViT-L/16`처럼 패치 크기를 함께 표기한다. 사전학습보다 높은 해상도로 파인튜닝할 때는 패치 크기를 그대로 두어 시퀀스가 길어지므로, 학습된 위치 임베딩을 원본 이미지에서의 위치에 맞춰 2D 보간해서 쓴다.

## 실험 결과

- **데이터 규모에 따른 역전**: ImageNet-1k만으로 학습하면 같은 크기의 ResNet보다 성능이 낮다. 귀납적 편향이 없는 만큼 작은 데이터에서는 불리하기 때문이다. 하지만 ImageNet-21k, JFT-300M처럼 데이터가 커질수록 ViT가 ResNet 계열(BiT)을 앞지른다.
- **최고 성능**: JFT-300M으로 사전학습한 `ViT-H/14`가 ImageNet top-1 약 88.5%를 기록해, 당시 최고 수준이던 BiT-L과 Noisy Student를 넘어섰다.
- **효율**: 같은 수준의 성능을 내는 데 필요한 사전학습 연산량이 BiT 대비 수 분의 일 수준이었다.
- **분석**: 학습된 위치 임베딩은 서로 가까운 패치끼리 유사도가 높은 2D 구조를 스스로 학습했고, 낮은 층부터 이미 이미지 전체에 걸친 attention을 하는 head가 존재했다.

## 느낀 점

- "이미지는 다르니까 이미지에 맞는 구조가 필요하다"는 상식을, 구조를 더하는 대신 **데이터를 늘려서** 뛰어넘었다는 점이 인상적이다. 귀납적 편향은 데이터가 적을 때는 도움이 되지만 충분히 클 때는 오히려 발목을 잡을 수 있다는 걸 정면으로 보여준 논문이다.
- 이 논문의 진짜 기여는 새로운 구조보다는 **NLP와 비전이 같은 백본을 공유할 수 있다**는 걸 보인 데 있다고 느꼈다. 이후 이미지 생성, 분할, 자기지도 학습 등 거의 모든 비전 연구가 ViT를 기본 백본으로 삼게 된 이유가 납득됐다.
- 단점도 분명하다. 패치 크기가 고정되어 세밀한 구조를 놓칠 수 있고, self-attention의 연산량이 시퀀스 길이의 제곱으로 늘어난다. 이후 Swin처럼 계층적 구조와 국소 attention을 되살린 후속 연구가 나온 배경이 이해됐다.
