---
layout: post
title: "[논문 리뷰] Flow Matching for Generative Modeling"
date: 2026-09-11 09:00:00+0900
description: 시뮬레이션 없이 continuous normalizing flow를 회귀 손실 하나로 직접 학습시키는 Flow Matching 논문 정리
tags: flow-matching normalizing-flow ode generative-model diffusion
categories: ["Generative AI"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Flow Matching for Generative Modeling
- **저자**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta AI / FAIR)
- **학회**: ICLR 2023

## 한 줄 요약

노이즈 분포에서 데이터 분포로 이동시키는 continuous normalizing flow(CNF)를, 미분방정식을 직접 풀거나(simulation) 시뮬레이션해서 최대우도를 역전파하는 대신, **각 시점에서의 속도장(vector field)을 회귀로 직접 맞추는 단순한 목적함수 하나로 학습**시키는 방법. [DDPM](/blog/2026/denoising-diffusion-probabilistic-models/) 같은 diffusion model을 특수한 경우로 포함하는 더 일반적인 프레임워크이면서, 학습이 훨씬 안정적이고 샘플링도 더 적은 스텝으로 가능하다.

## 배경

CNF는 원래 데이터 $x$를 노이즈 $x_0$까지 변형시키는 시간에 따른 흐름 $\phi_t(x_0)$을 상미분방정식(ODE)

$$
\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))
$$

으로 정의하고, 신경망으로 속도장 $v_t$를 파라미터화하는 생성모델이다. 이론적으로는 우아하지만, 학습하려면 매 스텝 ODE를 실제로 풀어서(numerical integration) 우도를 계산해야 했고, 이 시뮬레이션 비용이 커서 대규모 학습이 어려웠다.

반면 [DDPM](/blog/2026/denoising-diffusion-probabilistic-models/)류의 diffusion model은 forward process를 미리 고정해두고 "노이즈를 예측하는" 단순 회귀로 학습하기 때문에 시뮬레이션이 필요 없다. 하지만 diffusion은 forward process가 가우시안 노이즈 주입으로 한정되어 있어서, CNF가 원칙적으로 표현할 수 있는 임의의 확률 경로(probability path)를 다 쓰지 못한다.

Flow Matching은 이 둘 사이를 잇는다 — **diffusion처럼 시뮬레이션 없이 회귀로 학습**하면서도, **CNF처럼 임의의 확률 경로를 설계**할 수 있게 만든다.

## 제안 방법

### Marginal vs. Conditional 확률 경로

목표는 시간 $t=0$의 노이즈 분포 $p_0$에서 $t=1$의 데이터 분포 $p_1$로 가는 확률 경로 $p_t$를 만드는 속도장 $u_t$를 학습하는 것이다. 이상적인 손실은

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, x \sim p_t}\, \| v_t^\theta(x) - u_t(x) \|^2
$$

인데, 문제는 marginal 속도장 $u_t$ 자체를 직접 알 수 없다는 것이다(전체 데이터 분포에 대해 적분된 양이라서).

핵심 아이디어는, 데이터 샘플 $x_1$ **하나**가 주어졌을 때의 **조건부(conditional)** 확률 경로 $p_t(x \mid x_1)$와 그 조건부 속도장 $u_t(x \mid x_1)$은 훨씬 간단하게 설계할 수 있다는 점이다. 예를 들어 $x_1$에서 가우시안 노이즈로 가는 직선 경로를 쓰면 조건부 속도장은 닫힌 형태(closed form)로 바로 나온다. 논문은 marginal 속도장이 이 조건부 속도장들의 (데이터 분포에 대한) 가중 평균이라는 것을 보이고, **Conditional Flow Matching(CFM)** 손실

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t,\, x_1 \sim q(x_1),\, x \sim p_t(x \mid x_1)}\, \| v_t^\theta(x) - u_t(x \mid x_1) \|^2
$$

을 대신 최소화해도 $\mathcal{L}_{FM}$과 **같은 그래디언트**를 준다는 것을 증명한다. 즉, 모르는 marginal 속도장을 직접 다루는 대신, 데이터 샘플마다 알고 있는 조건부 속도장을 회귀 타겟으로 써서 학습하면 되는 것 — DDPM에서 "이번 스텝에 섞은 노이즈를 맞혀라"라는 타겟이 있었던 것과 정확히 같은 구조의 단순함이다.

### Optimal Transport 경로

조건부 경로를 어떻게 설계하느냐가 자유도인데, 논문은 노이즈와 데이터 사이를 **직선(straight-line)**으로 잇는 optimal-transport 경로를 제안한다.

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad u_t(x \mid x_1) = x_1 - x_0
$$

속도장이 시간에 대해 상수(직선의 방향 그대로)이기 때문에, diffusion의 곡선형 경로보다 ODE를 풀 때 필요한 스텝 수가 훨씬 적다. 논문은 이 경로가 diffusion의 가우시안 forward process로 유도되는 경로를 특수한 경우로 포함한다는 것도 보여서, **Flow Matching이 diffusion의 상위 프레임워크**임을 명확히 한다.

## 실험 결과

- ImageNet 32/64/128/256 등에서 학습된 CNF를, 기존 시뮬레이션 기반 CNF 학습법 및 diffusion 기반 방법들과 비교.
- 우도(negative log-likelihood)와 샘플 품질(FID) 모두에서 diffusion 기반 방법과 동등하거나 더 우수.
- optimal-transport 경로를 쓰면 ODE 궤적이 더 곧기 때문에, **훨씬 적은 함수 평가 횟수(NFE)로도** 비슷한 품질의 샘플을 생성 — 샘플링 효율이 diffusion 대비 크게 개선됨.

## 느낀 점

- [DDPM](/blog/2026/denoising-diffusion-probabilistic-models/)을 읽을 때 "왜 하필 가우시안 노이즈를 점진적으로 섞는 forward process였을까"가 다소 임의적으로 느껴졌는데, Flow Matching을 보고 나니 그게 **무수히 많은 가능한 확률 경로 중 하나**였을 뿐이라는 게 명확해졌다. "회귀로 학습 가능한 목적함수를 어떻게 유도하느냐"라는 관점에서 diffusion과 flow matching을 같은 틀로 이해할 수 있다는 게 이 논문의 가장 큰 정리 효과였다.
- Conditional Flow Matching이 marginal 목적함수와 같은 그래디언트를 준다는 증명이, VAE의 ELBO 유도나 diffusion의 변분 하한 유도와 결이 비슷하다 — "직접 못 다루는 양을 조건부로 쪼개서 다루기 쉬운 대리 목적함수로 바꾸되 그래디언트는 보존한다"는 패턴이 생성모델 전반에서 반복해서 등장하는 것 같다.
- 직선 경로 덕분에 적은 스텝으로 샘플링할 수 있다는 점은, Stable Diffusion 3 같은 최신 text-to-image 모델들이 왜 flow matching으로 갈아탔는지를 이해하는 데 직접적인 실마리가 됐다. 다음엔 이 아이디어가 실제 대규모 이미지 생성 모델에서 어떻게 쓰이는지를 더 찾아보고 싶다.
