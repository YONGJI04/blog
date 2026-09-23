---
layout: post
title: "[논문 리뷰] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
date: 2026-09-11 09:00:00+0900
description: 두 분포를 잇는 ODE의 궤적을 직선에 가깝게 반복 정제(reflow)해 적은 스텝으로 생성하는 Rectified Flow 논문 정리
tags: rectified-flow flow-matching ode generative-model reflow
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- **저자**: Xingchao Liu, Chengyue Gong, Qiang Liu (UT Austin)
- **학회**: ICLR 2023

## 한 줄 요약

두 분포 $\pi_0$(노이즈)와 $\pi_1$(데이터)를 잇는 ODE를, 두 샘플을 **직선으로 이은 경로의 속도**를 회귀하는 단순한 손실로 학습한다. 여기에 학습된 ODE로 만든 짝을 다시 학습에 쓰는 **reflow**를 반복하면 궤적이 점점 직선에 가까워져서, **한 스텝 또는 아주 적은 스텝**으로도 좋은 샘플을 만들 수 있다.

## 배경

[DDPM](/blog/2026/denoising-diffusion-probabilistic-models/) 같은 diffusion model은 품질이 좋지만 샘플링에 수십~수백 번의 네트워크 평가가 필요하다. 원인 중 하나는 생성 경로가 **곡선**이라는 점이다. ODE를 수치적으로 풀 때 곡선을 따라가려면 스텝을 잘게 나눠야 하고, 스텝을 줄이면 오차가 커진다.

그렇다면 애초에 경로가 직선이라면 ODE를 한 번의 큰 스텝으로 풀어도 오차가 없을 것이다. 이 논문은 "경로를 직선으로 만드는 방법"을 학습 문제로 정식화한다. 같은 시기에 나온 [Flow Matching](/blog/2026/flow-matching-for-generative-modeling/)과 같은 손실 형태에 도달하지만, 논문의 강조점은 **직선성(straightness)과 이를 개선하는 reflow**에 있다.

## 제안 방법

### Rectified Flow

$\pi_0$에서 뽑은 $X_0$와 $\pi_1$에서 뽑은 $X_1$을 짝지어 두고(처음에는 독립 커플링), 둘을 직선으로 잇는 보간 $X_t = t X_1 + (1-t) X_0$를 생각한다. 이 직선의 속도는 $X_1 - X_0$이다. 신경망 $v_\theta(x, t)$가 이 속도를 예측하도록 회귀한다.

$$
\min_\theta \; \mathbb{E}\left[\, \big\| (X_1 - X_0) - v_\theta(X_t, t) \big\|^2 \,\right]
$$

학습된 $v_\theta$로 ODE $dZ_t = v_\theta(Z_t, t)\,dt$를 $Z_0 \sim \pi_0$에서 풀면 $Z_1 \sim \pi_1$이 된다. 논문은 이 ODE가 **모든 시점 $t$에서 원래 직선 보간의 주변분포(marginal)를 그대로 보존**함을 보인다. 개별 직선 경로들은 서로 교차할 수 있지만, ODE는 교차점에서 경로를 갈아타는 식으로 이를 해소하기 때문에 궤적이 교차하지 않는 하나의 흐름이 된다.

### Reflow: 경로를 더 곧게 펴기

첫 번째 모델(1-Rectified Flow)의 ODE 궤적은 교차하는 직선들을 "풀어낸" 결과라서 아직 곡선이다. 이 모델로 $Z_0$에서 $Z_1$을 실제로 시뮬레이션해서 얻은 **새로운 짝** $(Z_0, Z_1)$을 다시 학습 데이터로 삼아 같은 손실을 최소화한다. 이 과정을 reflow라고 하고, 반복할수록 k-Rectified Flow의 궤적이 곧아진다.

논문은 reflow가 (1) 수송 비용(transport cost)을 볼록 비용 함수에 대해 늘리지 않고, (2) 궤적의 곧지 않은 정도를 나타내는 straightness 지표를 반복 횟수 $k$에 대해 $O(1/k)$로 줄인다는 이론적 결과를 보인다. 완전히 직선이 되면 한 스텝 오일러(Euler) 시뮬레이션이 정확하다.

### 증류와 도메인 변환

reflow로 직선에 가까워진 모델은 한 스텝 생성기로 **증류(distillation)**하기도 쉽다. 또한 $\pi_0$를 꼭 가우시안 노이즈로 둘 필요가 없어서, 고양이 이미지 분포에서 개 이미지 분포로 옮기는 식의 **이미지 간 변환(domain transfer)**에도 같은 틀을 그대로 쓸 수 있다.

## 실험 결과

- **CIFAR-10 무조건 생성**: 1-Rectified Flow를 정확한 ODE 솔버로 풀면 FID 약 2.6으로 당시 diffusion 계열과 견줄 만한 품질을 보였다.
- **적은 스텝**: reflow를 한 번 거친 2-Rectified Flow는 한 스텝 오일러만으로 FID 약 12를 기록해, 1-Rectified Flow의 한 스텝(수백대 FID)에서 크게 개선됐다. 여기에 증류를 더하면 한 스텝 FID가 약 4.9까지 내려갔다.
- 고해상도 이미지와 도메인 변환 실험에서도 reflow 횟수가 늘수록 궤적이 곧아지고, 적은 스텝에서의 품질이 좋아지는 경향을 보였다.

## 느낀 점

- [Flow Matching](/blog/2026/flow-matching-for-generative-modeling/)과 손실 형태가 사실상 같다는 점이 흥미로웠다. 같은 목적함수가 다른 동기로 거의 동시에 유도됐다는 것이고, 두 논문의 차이는 "손실을 어떻게 정당화하는가"보다 "그 위에서 무엇을 더 하는가"에 있다. Flow Matching이 확률 경로의 일반적 프레임워크라면, Rectified Flow는 **직선성**이라는 구체적인 목표를 정하고 reflow로 그 목표를 향해 반복 개선하는 절차까지 제시한다.
- 개별 학습 경로는 교차하지만 학습된 ODE는 교차하지 않는다는 점이 처음엔 직관적이지 않았다. 교차점에서 속도가 여러 방향의 **평균**이 되면서 경로들이 부드럽게 재배치된다는 설명을 읽고 나서야 이해가 됐고, 이게 곧 reflow가 필요한 이유이기도 하다.
- reflow는 생성 결과로 다시 학습한다는 점에서 비용이 들고 반복할수록 오차가 누적될 수 있다는 점이 아쉽다. 그럼에도 "궤적을 곧게 펴면 적은 스텝으로 풀린다"는 원리는 이후 대규모 text-to-image 모델들이 flow 기반 목적함수로 옮겨간 배경을 이해하는 데 핵심 단서가 됐다.
