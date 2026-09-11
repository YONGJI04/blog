---
layout: post
title: "[논문 리뷰] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields"
date: 2026-09-05 09:00:00+0900
description: 3D 가우시안을 표면에 붙는 2D 원반으로 눌러서 3DGS의 기하 정확도 문제를 해결한 2DGS 논문 정리
tags: 2dgs gaussian-splatting surface-reconstruction neural-rendering
categories: ["Computer Vision"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **저자**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao (ShanghaiTech University, University of Tübingen, Tübingen AI Center)
- **학회**: SIGGRAPH 2024 (ACM Transactions on Graphics)

## 한 줄 요약

[3D Gaussian Splatting](/blog/2026/3d-gaussian-splatting/)의 3차원 타원체 대신, 한 축의 스케일을 0으로 눌러 만든 **평평한 2차원 원반(disk) 형태의 가우시안**으로 장면을 표현한다. 3DGS는 렌더링 품질은 뛰어나지만 여러 시점에서 각 가우시안이 실제 표면과 정확히 일치하지 않아도 색만 맞으면 loss가 낮게 나오는 문제(멀티뷰 불일치)가 있는데, 2DGS는 가우시안 자체를 표면에 달라붙는 얇은 판으로 제약해서 이 문제를 구조적으로 없앤다. 그 결과 3DGS와 비슷한 실시간 렌더링 속도를 유지하면서 **훨씬 정확한 표면 형상(메쉬)**을 복원한다.

## 배경

[3D Gaussian Splatting](/blog/2026/3d-gaussian-splatting/)은 novel view synthesis 품질과 속도에서는 획기적이었지만, 복원된 3D 가우시안들로부터 깨끗한 표면(mesh)을 뽑으려고 하면 결과가 지저분했다. 원인은 3D 타원체 가우시안이 본질적으로 **부피(volume)를 가진 blob**이라는 데 있다 — 여러 시점에서 봤을 때 사진과 색이 맞기만 하면 되기 때문에, 실제 표면 위치와 어긋난 채로 두껍고 비스듬하게 놓여도 photometric loss는 낮게 나올 수 있다(multi-view inconsistency). 이 때문에 depth·normal을 뽑아보면 노이즈가 심하고, 표면 근처에서 가우시안들의 정렬이 시점마다 미묘하게 달라진다.

3D 재구성 결과를 CAD 모델과 정합(registration)하는 것처럼 정밀한 표면 비교가 필요한 작업에서는 점구름이나 메쉬가 실제 표면에 가깝게 정렬돼 있어야 하는데, 3DGS 원본은 이 용도에는 기하학적으로 부정확했다 — 2DGS가 풀려는 문제가 바로 이 지점이다.

## 제안 방법

### 표면 모델링: 3D 타원체 → 2D 원반

3DGS의 3D 공분산 $\Sigma = RSS^TR^T$에서 스케일 벡터 $S$의 한 성분을 0으로 고정한다. 그러면 타원체가 한 축 방향으로 완전히 눌린 **평평한 원반**이 되고, 눌린 방향이 곧 그 지점의 **표면 법선(normal)**이 된다. 즉 가우시안 하나하나가 "위치·크기·방향을 가진 작은 평면 조각"이 되어, 점구름이 아니라 **국소 평면(tangent plane)들의 집합**으로 장면 표면을 직접 표현하는 셈이다.

### 렌더링: Perspective-Accurate Splatting

3DGS는 3D 가우시안을 화면에 투영할 때 원근 변환을 국소적으로 아핀(affine) 근사해서 splat했는데, 이 근사가 카메라에 가깝거나 비스듬한 각도에서 오차를 키우는 원인 중 하나였다. 2DGS는 각 2D 가우시안이 놓인 평면과 카메라 광선의 **교차(ray-splat intersection)**를 직접 계산하는 방식으로 렌더링해서, 근사 없이 원근을 정확하게 반영한다. 이 정확한 교차 계산 덕분에 스케일이나 시점이 극단적이어도 splat 형태가 왜곡되지 않는다.

### 두 가지 정규화 손실

평평한 원반들이 실제로 일관된 표면을 이루도록, 학습에 두 가지 기하 정규화 항을 추가한다.

- **Depth Distortion loss**: 한 픽셀(광선)을 따라 누적되는 여러 가우시안들이 깊이 방향으로 너무 퍼지지 않고 좁게 모이도록 유도 — 한 광선 위에 있는 가우시안들이 실제 표면 한 지점 근처에 몰리게 만든다.
- **Normal Consistency loss**: 렌더링된(누적) normal map이, 깊이맵을 미분해서 얻은 normal과 일치하도록 강제 — 각 가우시안이 나타내는 법선과 실제 표면 형상에서 유도되는 법선이 서로 어긋나지 않게 맞춘다.

두 손실 모두 photometric loss와 함께 최적화되어, "사진과 색이 맞다"뿐 아니라 "여러 시점에서 봐도 같은 얇은 표면 위에 정렬돼 있다"는 제약을 추가로 건다.

## 실험 결과

- DTU, Tanks&Temples 벤치마크의 표면 복원 정확도(Chamfer distance)에서, 3DGS 기반 방식들은 물론 기존 implicit 방식(NeuS 등)과 비교해도 경쟁력 있는 정확도를 달성.
- 렌더링 화질(PSNR/SSIM)과 속도는 3DGS와 대등한 수준으로 유지 — "표면 정확도를 얻는 대가로 속도를 크게 희생하지 않는다."
- 정성적으로도 복원된 메쉬가 3DGS 대비 훨씬 매끈하고 노이즈가 적으며, 얇은 구조물이나 평평한 표면에서 특히 개선 폭이 크다.

## 느낀 점

- 3DGS 리뷰에서 "명시적 표현이라 편집이 쉽다"는 장점을 적었었는데, 2DGS를 보고 나니 명시적 표현이라도 **기하학적으로 무엇을 표현하도록 제약했는지**가 완전히 다른 문제라는 걸 알게 됐다. 3D blob에서 2D 원반으로 자유도를 하나 줄인 것뿐인데, 그 제약이 표면 정합이라는 다운스트림 작업의 정확도를 좌우한다는 점이 인상적이었다.
- 재구성 결과를 어떤 기준 형상과 정밀하게 비교·정합해야 하는 다운스트림 작업이라면, 3DGS의 blob 표현보다 2DGS처럼 표면에 정렬된 표현을 베이스로 고르는 편이 낫겠다는 생각이 들었다.
- Depth distortion loss와 normal consistency loss처럼, "렌더링 품질 손실 하나"가 아니라 **여러 기하학적 제약을 명시적으로 손실 항으로 추가**하는 패턴이 이후 3DGS 계열 후속 연구(SuGaR, GOF 등)에서도 반복해서 나타난다. 표면을 정확히 뽑고 싶을 때 어떤 종류의 정규화를 걸면 되는지에 대한 좋은 레퍼런스가 됐다.
