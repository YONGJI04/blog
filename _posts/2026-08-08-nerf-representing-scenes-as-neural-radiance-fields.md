---
layout: post
title: "[논문 리뷰] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
date: 2026-08-08 09:00:00+0900
description: 3D 장면을 MLP 하나로 표현하고 미분 가능한 볼륨 렌더링으로 새로운 시점의 이미지를 합성하는 NeRF 논문 정리
tags: nerf neural-rendering view-synthesis volume-rendering
categories: ["Computer Vision"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
- **저자**: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng (UC Berkeley, Google Research, UC San Diego)
- **학회**: ECCV 2020

## 한 줄 요약

장면(scene) 하나를 **좌표 → 색상/밀도**를 출력하는 MLP 하나로 통째로 표현하고, 카메라 광선을 따라 이 MLP를 여러 번 쿼리해 미분 가능한 볼륨 렌더링으로 합성한 뒤, 실제 촬영된 이미지와의 픽셀 차이만으로 학습하는 novel view synthesis 방법. 3D 형상에 대한 명시적 supervision 없이 2D 이미지들만으로 사진처럼 사실적인 새로운 시점 렌더링이 가능함을 보였고, 이후 [3D Gaussian Splatting](/blog/2026/3d-gaussian-splatting/)을 포함한 방대한 "radiance field" 연구 계열의 출발점이 되었다.

## 배경

새로운 시점의 이미지를 합성하는 novel view synthesis는 오래된 문제다. NeRF 이전에는 크게 두 갈래였다.

- **명시적 표현**: voxel grid, mesh, point cloud, multi-plane image(MPI) 등으로 장면을 저장. 해상도를 올리면 메모리가 3제곱으로 커지거나(voxel), 실제 표면이 아닌 영역은 표현하기 애매하다(mesh).
- **딥러닝 기반 이미지 합성**: CNN으로 새 시점 이미지를 직접 생성. 학습은 되지만 결과가 흐릿하거나 시점 간 일관성(view-consistency)이 떨어지는 경우가 많았다.

NeRF는 장면 자체를 **연속적인 5D 함수**로 정의해서 이 문제를 우회한다. 이산적인 격자 대신 임의의 3D 위치와 시선 방향에 대해 언제든 쿼리 가능한 함수를 MLP로 근사하는 것이다.

## 제안 방법

### 5D Radiance Field

3D 위치 $\mathbf{x} = (x,y,z)$와 시선 방향 $\mathbf{d} = (\theta, \phi)$를 입력받아, 그 지점의 색상 $\mathbf{c} = (r,g,b)$와 밀도(density) $\sigma$를 출력하는 함수를 MLP $F_\Theta$로 정의한다.

$$
F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)
$$

밀도 $\sigma$는 위치에만 의존하고(그 지점에 불투명한 물체가 있는지), 색상 $\mathbf{c}$는 위치와 시선 방향 모두에 의존한다(금속 표면의 반사광처럼 보는 각도에 따라 달라지는 색을 표현하기 위해).

### 볼륨 렌더링으로 이미지 합성

카메라에서 픽셀을 지나는 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 따라 $F_\Theta$를 여러 점에서 쿼리하고, 고전적인 볼륨 렌더링 적분식으로 최종 픽셀 색을 합성한다.

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt, \qquad T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)
$$

$T(t)$는 광선이 $t_n$에서 $t$까지 오는 동안 아무것도 없어서 투과(transmittance)될 확률이다. 실제 구현에서는 광선을 구간별로 나눠 적분을 이산합(quadrature)으로 근사한다. 이 전체 과정이 미분 가능하기 때문에, 렌더링된 픽셀과 실제 사진 픽셀의 MSE만으로 $F_\Theta$의 가중치를 역전파로 바로 학습할 수 있다 — **3D 형상에 대한 직접적인 라벨이 전혀 없어도** 여러 각도에서 찍은 2D 사진(과 알려진 카메라 pose)만으로 학습이 성립한다.

### Positional Encoding

$(\mathbf{x}, \mathbf{d})$를 MLP에 그대로 넣으면 신경망이 저주파(low-frequency) 성분에 편향되는 성질(spectral bias) 때문에 결과가 뭉개진다. NeRF는 입력을 고주파 sin/cos 함수로 먼저 매핑한 뒤 MLP에 넣는다.

$$
\gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right)
$$

같은 좌표를 여러 주파수의 sin/cos로 펼쳐서 넣는다는 점에서, [Transformer의 positional encoding](/blog/2026/attention-is-all-you-need/)과 수식 형태가 거의 동일하다 — 다만 목적은 다르다. Transformer는 순서 정보를 주입하는 것이고, NeRF는 MLP가 고주파 디테일(선명한 경계, 텍스처)을 표현할 수 있게 해주는 것이다.

### Hierarchical Sampling

광선 위 모든 지점을 균등하게 촘촘히 샘플링하면 낭비가 심하다(대부분의 지점엔 물체가 없다). NeRF는 "coarse" 네트워크로 광선을 성글게 훑어 밀도가 높은 구간을 대략 파악한 뒤, 그 구간에 샘플을 집중시켜 "fine" 네트워크를 다시 쿼리하는 2단계 샘플링을 쓴다. 계산을 표면 근처에 집중시켜 효율을 높이는 방식이다.

## 실험 결과

- 합성 장면(Blender 렌더링)과 실제 촬영 장면(LLFF, 핸드헬드 카메라로 찍은 forward-facing 장면) 모두에서 당시 SOTA였던 Neural Volumes, Scene Representation Networks(SRN), Local Light Field Fusion(LLFF) 대비 큰 폭으로 향상된 PSNR/SSIM을 달성.
- 반사광, 굴절 같은 view-dependent 효과와 얇은 구조물(나뭇가지 등)까지 사실적으로 재현.
- 단점도 명확: 장면 하나당 학습에 수 시간~하루가 걸리고, 렌더링도 광선 하나당 수백 번씩 MLP를 쿼리해야 해서 이미지 한 장 만드는 데 수 초가 걸린다. 실시간 렌더링과는 거리가 멀다.

## 느낀 점

- "장면 = MLP 하나"라는 프레이밍이 우아하다. voxel처럼 메모리를 명시적으로 잡아먹지 않고, 연속 함수라서 이론적으로 해상도 제한이 없다.
- 하지만 이 우아함의 대가가 바로 속도다. 픽셀 하나, 광선 하나마다 MLP를 수백 번 순전파해야 하니, 파라미터 자체는 적어도 렌더링 연산량이 어마어마하다. GAN/VAE/DDPM 리뷰에서 "학습 안정성 대 생성 품질 대 샘플링 속도"라는 삼각 트레이드오프를 계속 봤는데, NeRF는 "표현력 대 렌더링 속도" 축에서 명확히 표현력 쪽에 치우쳐 있다.
- 이 속도 문제를 정면으로 공략한 게 바로 다음에 볼 3D Gaussian Splatting이다. NeRF가 암묵적(implicit) MLP + ray marching이라면, 3DGS는 명시적(explicit) 3D 점들 + rasterization으로 완전히 다른 길을 택해서 같은 목표(사실적인 novel view synthesis)를 실시간으로 풀어낸다.
