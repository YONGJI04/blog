---
layout: post
title: "[논문 리뷰] DINOv2: Learning Robust Visual Features without Supervision"
date: 2026-07-29 09:00:00+0900
description: 큐레이션한 1.4억 장 데이터와 자기지도 학습으로 파인튜닝 없이 쓸 수 있는 범용 시각 특징을 만든 DINOv2 논문 정리
tags: dinov2 self-supervised-learning vit foundation-model representation-learning
categories: ["Computer Vision"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: DINOv2: Learning Robust Visual Features without Supervision
- **저자**: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski (Meta AI Research)
- **게재**: TMLR 2024

## 한 줄 요약

라벨이나 텍스트 없이 이미지만으로 학습한 [ViT](/blog/2026/an-image-is-worth-16x16-words/)가, **파인튜닝 없이 고정(frozen)한 채로 분류·분할·깊이 추정·검색 등 다양한 태스크에 바로 쓸 수 있는** 범용 특징을 내놓도록 만든 논문이다. 새로운 학습 기법을 하나 제안하기보다, 기존 자기지도 기법들을 결합하고 **데이터 큐레이션과 대규모 학습 안정화**를 정교하게 다듬어 자기지도 학습이 CLIP류의 약한 지도(weakly-supervised) 특징에 필적할 수 있음을 보였다.

## 배경

NLP에서는 대규모 텍스트로 사전학습한 모델을 파인튜닝 없이 여러 태스크에 쓰는 파운데이션 모델이 일반화되었다. 비전에서는 이미지-텍스트 쌍으로 학습한 CLIP 계열이 그 역할을 했지만, 텍스트 캡션은 이미지의 세부 정보(위치, 깊이, 국소 구조)를 충분히 담지 못한다는 한계가 있다.

자기지도 학습은 캡션 없이 이미지 자체에서 특징을 배우므로 이런 정보를 더 잘 보존할 수 있다. 다만 기존 자기지도 연구는 주로 ImageNet 같은 작고 정제된 데이터에서 이루어졌고, 데이터를 크게 늘리면 품질이 오히려 떨어지는 문제가 있었다. DINOv2는 **"충분히 크고 잘 정제된 데이터 + 안정적인 대규모 학습"**으로 이 한계를 넘어선다.

## 제안 방법

### 데이터 큐레이션: LVD-142M

웹에서 수집한 약 12억 장의 정제되지 않은 이미지에서, 중복을 제거하고 ImageNet-22k 같은 정제된 데이터셋과 **임베딩이 비슷한 이미지를 검색(retrieval)해 골라내는** 방식으로 1.42억 장의 LVD-142M 데이터셋을 만든다. 라벨이나 메타데이터를 쓰지 않고 이미지 임베딩만으로 큐레이션하기 때문에 자기지도 학습의 전제도 지킨다.

### 학습 목적함수

교사(teacher)-학생(student) 자기증류 구조를 기반으로 두 가지 손실을 함께 쓴다. 교사는 학생의 지수이동평균(EMA)으로 갱신된다.

- **이미지 수준 손실 (DINO)**: 같은 이미지의 서로 다른 크롭에서 나온 `[CLS]` 토큰 표현이 교사·학생 사이에 일치하도록 학습한다.
- **패치 수준 손실 (iBOT)**: 학생에게 입력 패치 일부를 마스킹해서 주고, 마스킹된 위치의 패치 표현을 교사의 표현에 맞추도록 학습한다. 국소적인 정보를 배우는 데 중요하다.

여기에 학습 안정성과 특징 품질을 높이는 여러 세부 기법을 더한다.

- 이미지 수준과 패치 수준 head의 가중치를 분리(untying)
- 표현 붕괴를 막는 Sinkhorn-Knopp 센터링
- 배치 안에서 특징이 고르게 퍼지도록 하는 KoLeo 정규화
- 학습 마지막에 짧게 해상도를 높여(518px) 세밀한 예측을 위한 적응

### 효율적인 대규모 학습

FlashAttention 기반의 빠른 attention, 서로 다른 크롭을 한 시퀀스로 묶어 처리하는 sequence packing, 확률적 깊이(stochastic depth)의 효율적 구현, FSDP를 이용한 분산 학습 등으로 10억 파라미터급 ViT-g/14 학습을 가능하게 한다.

### 증류

가장 큰 ViT-g/14를 교사로 삼아 더 작은 ViT-S/B/L을 증류(distillation)로 학습한다. 작은 모델을 처음부터 학습하는 것보다 성능이 좋아서, 추론 비용이 작은 모델도 큰 모델의 특징 품질을 상당 부분 물려받는다.

## 실험 결과

- **선형 분류**: 파인튜닝 없이 고정된 특징 위에 선형 분류기만 얹은 ImageNet-1k 평가에서 ViT-g/14가 86%대의 top-1 정확도를 기록해, 기존 자기지도 방법들을 크게 앞서고 OpenCLIP 등 약한 지도 방법과 대등하거나 더 나은 성능을 보였다.
- **밀집 예측**: 의미 분할과 단안 깊이 추정처럼 국소 정보가 중요한 태스크에서 고정된 특징 + 단순한 head만으로 강한 성능을 냈다. 이미지-텍스트로 학습한 특징이 상대적으로 약한 영역이다.
- **인스턴스 검색, 비디오, 세밀 분류**: 별도 학습 없이도 여러 벤치마크에서 좋은 성능을 보여 "하나의 특징으로 여러 태스크"라는 범용성을 확인했다.
- **정성적 분석**: 특징을 PCA로 시각화하면 서로 다른 객체 사이에서도 같은 부위(예: 여러 동물의 머리, 다리)가 비슷한 색으로 나타나, 라벨 없이 의미적 대응 관계를 학습했음을 볼 수 있다.

## 느낀 점

- 새로운 손실이나 구조보다 **"어떤 데이터로, 어떻게 안정적으로 학습시키는가"**가 결과를 가른다는 점이 가장 인상 깊었다. 자기지도 학습이 스케일에서 밀린 것이 아니라, 데이터 품질과 학습 안정화를 제대로 다루지 못해서였다는 주장이 설득력 있게 보였다.
- CLIP이 이미지-텍스트 정렬을 통해 전역적인 의미를 잡는다면, DINOv2는 이미지 자체에서 국소 구조까지 잡는다는 대비가 명확했다. 분할이나 깊이처럼 위치 정보가 필요한 태스크에서 고정 백본으로 DINOv2가 널리 쓰이는 이유가 납득됐다.
- 데이터 큐레이션에 여전히 정제된 데이터셋(ImageNet-22k 등)이 "씨앗"으로 쓰인다는 점은 순수한 자기지도의 한계로도 읽혔다. 큐레이션 자체를 라벨 없이 어디까지 자동화할 수 있는지가 후속 연구의 관건일 것 같다.
