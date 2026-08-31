---
layout: post
title: "[논문 리뷰] 3D Gaussian Splatting for Real-Time Radiance Field Rendering"
date: 2026-08-28 09:00:00+0900
description: 3D 장면을 명시적인 3D 가우시안들로 표현하고 rasterization으로 실시간 렌더링하는 3DGS 논문 정리
tags: 3dgs gaussian-splatting neural-rendering real-time-rendering
categories: ["Computer Vision"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **저자**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria, Université Côte d'Azur, Max Planck Institute for Informatics)
- **학회**: SIGGRAPH 2023 (ACM Transactions on Graphics)

## 한 줄 요약

[NeRF](/blog/2026/nerf-representing-scenes-as-neural-radiance-fields/)처럼 MLP와 ray marching으로 장면을 암묵적으로 표현하는 대신, 장면을 수백만 개의 **명시적인 3D 가우시안**(위치·모양·색·불투명도를 가진 점 같은 원시 도형)으로 표현하고, 이를 화면에 직접 **rasterize(splat)**해서 렌더링한다. MLP 순전파를 아예 없애버려서 NeRF급 화질을 유지하면서도 **1080p에서 실시간(30~100+fps) 렌더링**을 처음으로 달성했다.

## 배경

NeRF 이후 Instant-NGP, Plenoxels 같은 후속 연구들이 해시 그리드나 명시적 voxel로 학습/렌더링 속도를 크게 개선했지만, 여전히 픽셀마다 광선을 쏘고 그 위의 여러 지점을 순회하며 값을 누적하는 **ray marching** 구조 자체는 유지했다. 이 구조는 태생적으로 픽셀 수 × 광선당 샘플 수만큼의 연산이 필요해서, 아무리 개별 쿼리를 가볍게 만들어도 진짜 실시간(60fps급) 고화질 렌더링에는 한계가 있었다.

3DGS는 질문 자체를 바꾼다: "광선을 따라가며 뭐가 있는지 찾는" 대신, "장면을 이루는 물체들을 화면에 그대로 투영해서 그리면 안 되나?" — 그래픽스에서 오래된 **rasterization**(폴리곤을 화면에 투영해 그리는 방식)의 아이디어를 점 기반 표현에 맞게 되살린 것이다.

## 제안 방법

### 장면 표현: 이방성 3D 가우시안

장면을 수십만~수백만 개의 3D 가우시안 집합으로 표현한다. 가우시안 하나는 다음 파라미터를 가진다.

- 위치(평균) $\boldsymbol{\mu}$
- 공분산 $\Sigma$ — 가우시안의 크기와 방향(회전)을 결정. 학습 안정성을 위해 직접 최적화하지 않고, scale 벡터 $s$와 회전 quaternion $q$로 분해해서 최적화한다.
  $$
  \Sigma = R S S^T R^T
  $$
  이렇게 하면 $\Sigma$가 항상 양의 준정부호(positive semi-definite) 행렬이라는 물리적으로 유효한 형태를 자동으로 보장한다.
- 불투명도(opacity) $\alpha$
- 색상 — spherical harmonics(SH) 계수로 표현해서, NeRF의 시선 방향 의존성과 마찬가지로 **보는 각도에 따라 달라지는 색**(반사광 등)을 표현할 수 있다.

즉 point cloud에 각 점마다 "퍼짐 정도와 방향"을 가진 3D 타원체를 붙인 형태로, 표면을 여러 개의 부드러운 얼룩(blob)들로 근사하는 셈이다.

### 렌더링: Differentiable Splatting

각 3D 가우시안을 카메라 시점으로 투영(project)하면 2D 화면 공간에서도 가우시안 형태가 된다(원근 투영의 국소 아핀 근사, EWA splatting 기법). 이 2D 가우시안들을 카메라로부터의 깊이 순으로 정렬한 뒤, 픽셀별로 알파 블렌딩한다.

$$
C = \sum_{i} c_i\, \alpha_i \prod_{j<i} (1-\alpha_j)
$$

식 자체는 NeRF의 볼륨 렌더링과 형태가 비슷하지만(앞쪽 가우시안이 뒤쪽을 가리는 정도를 누적), 핵심 차이는 **연속 적분을 근사하기 위해 광선마다 MLP를 수백 번 쿼리하는 대신, 정렬된 이산적인 가우시안들을 GPU 래스터라이저로 한 번에 그린다**는 점이다. 논문은 이 전체 파이프라인을 위한 커스텀 CUDA 타일 기반 래스터라이저를 직접 구현해서, 정렬·블렌딩·역전파(gradient)까지 전부 GPU에서 고속으로 처리한다.

### 최적화와 Adaptive Density Control

초기값은 NeRF와 동일하게 SfM(COLMAP)으로 얻은 sparse point cloud에서 시작한다. 이후 실제 이미지와 렌더링 결과의 photometric loss를 래스터라이저를 통해 각 가우시안 파라미터(위치·공분산·불투명도·SH 색상)로 역전파해서 경사하강으로 최적화한다.

여기에 더해, 학습 중 주기적으로 가우시안 개수 자체를 조정하는 휴리스틱을 적용한다.

- **Clone**: gradient가 크고 가우시안이 작은(디테일이 부족한) 영역 → 복제해서 밀도를 높임
- **Split**: gradient가 크고 가우시안이 큰(과대표현된) 영역 → 더 작은 가우시안 여러 개로 분할
- **Prune**: 불투명도가 거의 0이거나 지나치게 큰 가우시안 → 제거

시작은 성긴 point cloud였지만, 학습이 진행되며 필요한 곳에 가우시안이 자동으로 늘어나고 불필요한 곳은 정리되는 구조다.

## 실험 결과

- Mip-NeRF360, Tanks&Temples 등 복잡한 실제 360도 장면 데이터셋에서, 화질(PSNR/SSIM/LPIPS)은 당시 최고 품질이던 Mip-NeRF360과 동등하거나 상회.
- 렌더링 속도는 압도적: 1080p 기준 초당 수십~100+ 프레임으로, 같은 화질의 NeRF 계열보다 수백 배 빠름.
- 학습 시간도 Instant-NGP, Plenoxels 같은 고속 NeRF 변형과 대등하거나 더 빠름 — 화질·학습속도·렌더링속도 세 축을 동시에 만족시킨 첫 사례로 평가받는다.

## 느낀 점

- NeRF 리뷰 말미에 적었던 "표현력 대 렌더링 속도" 트레이드오프를, 3DGS는 **암묵적 표현을 명시적 표현으로 바꾸는 것**으로 정면 돌파했다. MLP 순전파를 아예 없애고 GPU가 원래 잘하는 rasterization 파이프라인에 태워버린 발상이 인상적이다 — "새로운 알고리즘"이 아니라 "이미 존재하는 최적화된 하드웨어 파이프라인에 문제를 맞춰 넣은 것"에 가깝다.
- 명시적 표현이라는 점은 속도 외에도 실용적 이점이 크다. 가우시안 하나하나가 위치·크기를 가진 실체이므로 직접 옮기거나 지울 수 있어서, MLP 가중치라는 블랙박스보다 장면 편집(editing)이 훨씬 쉽다. 3D 정합/재구성 작업에서 point cloud를 다루는 것과 자연스럽게 이어지는 지점이라 흥미로웠다.
- 단점도 명확하다: 복잡한 장면은 가우시안이 수백만 개까지 늘어나 저장 용량이 크고, clone/split/prune 임계값 같은 휴리스틱이 손으로 튜닝된 부분이라 NeRF의 깔끔한 확률적 볼륨 렌더링 공식화에 비하면 다소 "공학적"이다. 그럼에도 실시간이라는 실용적 이득이 워낙 커서, 이후 4D(동적 장면), SLAM, 아바타 등 거의 모든 방향으로 확장 연구가 쏟아진 이유가 납득이 갔다.
