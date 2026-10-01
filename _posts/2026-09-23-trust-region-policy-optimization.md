---
layout: post
title: "[논문 리뷰] Trust Region Policy Optimization (TRPO)"
date: 2026-09-23 09:00:00+0900
description: 정책 업데이트 폭을 KL 발산으로 묶어 성능이 단조롭게 개선되도록 보장한 TRPO 논문 정리
tags: trpo reinforcement-learning policy-gradient trust-region kl-divergence
categories: ["Reinforcement Learning"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Trust Region Policy Optimization
- **저자**: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel (UC Berkeley)
- **학회**: ICML 2015

## 한 줄 요약

정책을 한 번에 너무 많이 바꾸면 성능이 무너진다는 정책 경사(policy gradient)의 고질적인 문제를, **새 정책과 이전 정책 사이의 KL 발산이 일정 범위(trust region)를 넘지 않도록 제약**을 건 최적화로 푼 논문. 이론적으로는 매 업데이트마다 성능이 떨어지지 않는(단조 개선) 하한을 유도하고, 실제로는 이를 근사해 큰 신경망 정책도 안정적으로 학습할 수 있게 했다.

## 배경

정책 경사 방법은 기대 보상 $\eta(\pi)$를 정책 파라미터 $\theta$에 대해 경사 상승한다. 문제는 **스텝 크기**다. 파라미터 공간에서의 작은 변화가 정책(행동 분포)에서는 큰 변화일 수 있고, 한 번 나쁜 정책으로 가면 그 정책이 모은 나쁜 데이터로 다시 학습하게 되어 회복이 어렵다. 지도학습과 달리 강화학습은 **데이터 분포 자체가 현재 정책에 의존**하기 때문에, 잘못된 한 스텝이 학습 전체를 무너뜨릴 수 있다.

Kakade & Langford(2002)의 conservative policy iteration은 이전 정책과 새 정책을 섞는 방식으로 개선 하한을 보였지만, 혼합 정책이라는 형태가 실제 신경망 정책에 쓰기 어려웠다. TRPO는 이 이론을 일반적인 확률적 정책으로 확장하는 데서 출발한다.

## 제안 방법

### 대리 목적함수와 개선 하한

새 정책 $\tilde\pi$의 성능은 이전 정책 $\pi$의 advantage로 정확히 쓸 수 있다.

$$
\eta(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\tilde\pi}(s) \sum_a \tilde\pi(a \mid s) A_\pi(s, a)
$$

여기서 $\rho_{\tilde\pi}$는 새 정책이 방문하는 상태 분포인데, 이건 새 정책으로 실제로 돌려보기 전에는 알 수 없다. 그래서 이를 **이전 정책의 상태 분포 $\rho_\pi$로 바꾼 근사** $L_\pi(\tilde\pi)$를 대리 목적함수(surrogate objective)로 쓴다. 두 정책이 가까우면 이 근사는 정확하다.

논문의 핵심 정리는 이 근사 오차의 상한이다.

$$
\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C \cdot D_{KL}^{\max}(\pi, \tilde\pi), \quad C = \frac{4 \epsilon \gamma}{(1-\gamma)^2}
$$

오른쪽을 최대화하면 실제 성능 $\eta$가 절대 떨어지지 않는다(minorization-maximization). 즉 **"대리 목적함수는 올리되, 이전 정책에서 너무 멀어지면 벌점"** 이라는 구조가 이론적으로 정당화된다.

### 실용적인 근사: KL 제약 최적화

이론의 벌점 계수 $C$를 그대로 쓰면 스텝이 지나치게 작아진다. 그래서 벌점 대신 **제약 조건**으로 바꾸고, 모든 상태에서의 최대 KL 대신 **평균 KL**을 쓴다.

$$
\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_\text{old}}(a \mid s)} A_{\theta_\text{old}}(s, a)\right] \quad \text{s.t.} \quad \mathbb{E}\left[D_{KL}(\pi_{\theta_\text{old}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s))\right] \le \delta
$$

목적함수의 확률비(importance ratio) 덕분에 이전 정책으로 모은 데이터로 새 정책을 평가할 수 있다.

### 풀이: 켤레 기울기와 line search

이 제약 최적화는 목적함수를 1차, KL 제약을 2차(Fisher 정보 행렬 $F$)로 근사해서 푼다. 업데이트 방향은 $F^{-1} g$로, 결국 **자연 경사(natural gradient)** 방향이다. 파라미터가 수만 개인 신경망에서 $F$를 직접 만들고 역행렬을 구할 수는 없으므로,

- **켤레 기울기법(conjugate gradient)** 으로 $F x = g$를 풀되, $F$ 자체 대신 Fisher-벡터 곱만 계산하고,
- 구한 방향으로 최대 스텝을 잡은 뒤 **backtracking line search**로 실제 KL 제약을 만족하고 목적함수가 개선되는지 확인하며 스텝을 줄인다.

데이터 수집은 궤적을 그대로 쓰는 **single path** 방식과, 상태마다 여러 행동을 굴려보는 **vine** 방식 두 가지를 제시한다.

## 실험 결과

- MuJoCo 시뮬레이션의 Swimmer, Hopper, Walker 보행 과제에서 신경망 정책을 처음부터 학습해, 자연 경사·CEM·CMA 등 기존 방법보다 안정적이고 좋은 성능을 냈다. 특히 Hopper와 Walker처럼 넘어지기 쉬운 과제에서 차이가 컸다.
- 같은 알고리즘을 그대로 Atari 게임에 적용해, 화면 픽셀을 입력으로 받는 합성곱 정책도 학습할 수 있음을 보였다. 일부 게임에서는 당시 DQN 계열과 견줄 만한 점수를 냈다.
- 과제마다 하이퍼파라미터를 거의 바꾸지 않고도 동작했다는 점이 강조된다. 스텝 크기를 KL 범위 $\delta$ 하나로 정하니 과제가 바뀌어도 튜닝 부담이 적었다.

## 느낀 점

- 정책 경사의 학습률 문제를 "파라미터를 얼마나 움직일까"가 아니라 **"행동 분포를 얼마나 바꿀까"** 로 다시 정의한 것이 핵심이라고 느꼈다. 파라미터 공간의 거리는 의미가 불분명하지만, KL은 정책이 실제로 얼마나 달라졌는지를 직접 잰다.
- 이론의 하한을 그대로 쓰면 너무 보수적이라 평균 KL 제약으로 완화하는 과정이, 이론과 실용 사이의 타협을 보여주는 좋은 예 같았다. 이론이 방향을 정하고, 실제 알고리즘은 그 방향을 유지한 채 계수를 현실적으로 바꾼다.
- 켤레 기울기와 line search 때문에 구현이 복잡하고 계산도 무겁다는 게 분명한 한계다. 바로 이 부분을 1차 최적화만으로 단순하게 만든 것이 다음에 볼 [PPO](/blog/2026/proximal-policy-optimization-algorithms/)이고, "이전 정책에서 너무 멀어지지 말라"는 아이디어는 지금의 LLM 강화학습까지 그대로 이어진다.
