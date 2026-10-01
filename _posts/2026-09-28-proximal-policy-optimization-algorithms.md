---
layout: post
title: "[논문 리뷰] Proximal Policy Optimization Algorithms (PPO)"
date: 2026-09-28 09:00:00+0900
description: TRPO의 KL 제약을 확률비 클리핑으로 대신해 1차 최적화만으로 안정적인 정책 학습을 가능하게 한 PPO 논문 정리
tags: ppo reinforcement-learning policy-gradient clipping actor-critic
categories: ["Reinforcement Learning"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: Proximal Policy Optimization Algorithms
- **저자**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
- **발표**: arXiv 2017

## 한 줄 요약

[TRPO](/blog/2026/trust-region-policy-optimization/)처럼 정책이 한 번에 너무 많이 바뀌지 않게 하되, 복잡한 KL 제약 최적화 대신 **새 정책과 이전 정책의 확률비를 일정 범위로 잘라내는(clipping) 간단한 목적함수**를 쓰는 방법. 일반적인 SGD/Adam으로 같은 데이터를 여러 번 재사용해 학습할 수 있어서, 구현이 쉽고 성능도 TRPO와 비슷하거나 더 좋다.

## 배경

정책 경사 방법에서 이전 정책으로 모은 데이터를 여러 번 재사용하면 표본 효율이 좋아지지만, 그대로 여러 epoch 최적화하면 정책이 데이터를 모은 정책에서 너무 멀어져 학습이 망가진다. TRPO는 이를 KL 제약으로 막았지만, 켤레 기울기와 line search가 필요해서 구현이 복잡하고, 드롭아웃이나 정책·가치 함수의 파라미터 공유 같은 구조와 함께 쓰기도 어려웠다.

PPO는 "TRPO의 안정성은 유지하면서 1차 최적화만으로 끝낼 수는 없을까"라는 질문에서 출발한다.

## 제안 방법

### 확률비와 클리핑 목적함수

확률비를 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$로 두면, TRPO의 대리 목적함수는 $\mathbb{E}_t[r_t(\theta) \hat A_t]$이다. 제약 없이 이것만 최대화하면 $r_t$가 한없이 커지는 방향으로 정책이 움직인다.

PPO는 여기에 클리핑을 건다.

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat A_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\right)\right]
$$

논문의 기본값은 $\epsilon = 0.2$이다. 동작을 경우별로 보면 직관적이다.

- **advantage가 양수**(좋은 행동): 그 행동의 확률을 올리되, $r_t$가 $1+\epsilon$을 넘으면 더 올려도 이득이 없다.
- **advantage가 음수**(나쁜 행동): 그 행동의 확률을 내리되, $r_t$가 $1-\epsilon$ 아래로 가면 더 내려도 이득이 없다.
- $\min$을 취하기 때문에, 정책이 **오히려 나빠지는 방향**으로 벗어난 경우에는 클리핑되지 않은 값이 선택되어 그대로 벌점을 받는다. 즉 목적함수는 원래 목적함수의 비관적인 하한이 된다.

결과적으로 이전 정책 주변에서만 개선 이득을 주는 셈이라, KL 제약 없이도 업데이트 폭이 자연스럽게 제한된다.

### KL 벌점 버전

비교용으로, KL을 제약 대신 벌점 $\beta \cdot \text{KL}$로 목적함수에 넣고, 측정된 KL이 목표값보다 크면 $\beta$를 키우고 작으면 줄이는 **적응형 KL 벌점** 버전도 제시한다. 실험에서는 클리핑 버전이 더 좋았다.

### 전체 학습 루프

실제 알고리즘은 actor-critic 구조다.

$$
L(\theta) = \mathbb{E}_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\right]
$$

정책 손실에 가치 함수 손실 $L^{VF}$과 탐색을 위한 엔트로피 보너스 $S$를 더한다. 학습은

1. $N$개의 병렬 actor가 각각 $T$ 스텝씩 환경을 진행해 데이터를 모으고,
2. advantage를 GAE(generalized advantage estimation)로 계산한 뒤,
3. 모은 데이터로 미니배치 SGD를 **$K$ epoch** 반복하는

과정을 되풀이한다. 같은 데이터를 여러 epoch 재사용할 수 있다는 점이 바닐라 정책 경사 대비 큰 장점이다.

## 실험 결과

- MuJoCo 연속 제어 7개 과제에서 클리핑 없음, 고정 KL 벌점, 적응형 KL 벌점, 여러 $\epsilon$ 값의 클리핑을 비교했다. **$\epsilon = 0.2$ 클리핑이 평균 점수가 가장 높았고**, 클리핑이 없는 버전은 일부 과제에서 오히려 시작 정책보다 나빠졌다.
- 같은 과제에서 TRPO, CEM, A2C 등과 비교해 대부분의 과제에서 가장 좋거나 비슷한 성능을 냈다.
- Atari 49개 게임에서 A2C, ACER와 비교했다. 학습 전체 평균 보상 기준으로는 PPO가 가장 많은 게임에서 이겼고, 마지막 100 에피소드 기준으로는 ACER와 비슷한 수준이었다. 구현은 ACER보다 훨씬 단순하다.
- 3D 휴머노이드가 달리고 방향을 바꾸는 고차원 연속 제어 과제도 학습할 수 있음을 보였다.

## 느낀 점

- TRPO의 이론적인 KL 제약을 **"확률비가 범위를 벗어나면 기울기를 끊는다"** 는 한 줄짜리 규칙으로 바꿨는데도 성능이 유지된다는 점이 인상 깊었다. 이론적 보장은 약해졌지만, 실제로 쓰이는 알고리즘은 이렇게 단순한 쪽이 이긴다는 걸 보여준다.
- $\min$과 clip의 조합이 처음엔 헷갈렸는데, advantage 부호별로 나눠 보니 "좋아지는 방향으로는 일정 이상 이득을 주지 않고, 나빠지는 방향으로는 끝까지 벌점을 준다"는 비대칭 구조라는 게 이해됐다.
- PPO는 이후 InstructGPT의 RLHF 단계에서 언어 모델을 학습하는 표준 알고리즘이 되었다. 다만 LLM에서는 정책만큼 큰 가치 함수(critic)를 따로 학습해야 해서 메모리 부담이 크고, 바로 이 critic을 없애고 그룹 내 상대 보상으로 advantage를 대신하는 방향이 이후 GRPO로 이어진다.
