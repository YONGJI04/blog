---
layout: post
title: "[논문 리뷰] DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)"
date: 2026-10-02 09:00:00+0900
description: 가치 함수 없이 그룹 내 상대 보상으로 advantage를 계산하는 GRPO를 제안하고, 수학 추론 특화 7B 모델을 만든 DeepSeekMath 논문 정리
tags: grpo deepseekmath reinforcement-learning llm math-reasoning ppo
categories: ["Reinforcement Learning"]
related_posts: false
toc:
  sidebar: left
---

- **논문**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **저자**: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo (DeepSeek-AI, Tsinghua University, Peking University)
- **발표**: arXiv 2024

## 한 줄 요약

웹에서 수학 데이터 120B 토큰을 골라내 7B 모델을 수학에 특화시키고, 그 위에 **PPO에서 가치 함수(critic)를 없앤 강화학습 알고리즘 GRPO(Group Relative Policy Optimization)** 를 적용해 수학 추론 성능을 끌어올린 논문. 같은 질문에 대해 여러 답을 샘플링하고, **그 그룹 안에서의 상대적인 보상**을 advantage로 쓰는 것이 핵심이다.

## 배경

LLM을 강화학습으로 다듬을 때 표준은 [PPO](/blog/2026/proximal-policy-optimization-algorithms/)였다. PPO는 advantage를 계산하려고 정책 모델과 비슷한 크기의 **가치 함수 모델**을 함께 학습한다. LLM에서는 이게 큰 부담이다. 메모리와 계산이 거의 두 배로 들고, 보상은 보통 답변 끝에 한 번만 주어지는데 가치 함수는 토큰마다 정확한 값을 내야 해서 학습 자체도 어렵다.

또 하나의 문제는 데이터다. 공개된 수학 사전학습 데이터는 규모가 작아서, 오픈 모델의 수학 실력이 GPT-4 같은 비공개 모델에 크게 뒤처져 있었다.

## 제안 방법

### DeepSeekMath Corpus

OpenWebMath를 시드로 fastText 분류기를 학습해 Common Crawl에서 수학 관련 웹페이지를 골라낸다. 골라낸 페이지가 많은 도메인을 찾아 사람이 수학 관련 URL을 추가로 표시하고, 이를 다시 분류기 학습에 넣는 과정을 여러 번 반복한다. 이렇게 **약 120B 토큰**의 수학 말뭉치를 만들었다.

DeepSeek-Coder-Base-v1.5 7B에서 출발해 이 말뭉치를 포함한 데이터로 500B 토큰을 추가 학습한 것이 DeepSeekMath-Base 7B다. 코드로 학습된 모델에서 시작하는 편이 수학 추론에 더 유리했다는 점도 보고한다.

### GRPO: 가치 함수 없는 PPO

질문 $q$ 하나에 대해 이전 정책으로 답변 $G$개 $\{o_1, \dots, o_G\}$를 샘플링하고, 보상 모델로 각각의 보상 $r_i$를 매긴다. advantage는 가치 함수 대신 **그룹 안에서 정규화한 보상**으로 계산한다.

$$
\hat A_{i,t} = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}
$$

답변 끝에 보상이 한 번 주어지는 경우(outcome supervision), 이 값을 답변의 모든 토큰에 똑같이 준다. 같은 질문에 대한 다른 답들보다 나으면 강화되고, 못하면 억제되는 구조다. 보상 모델이 원래 "답들끼리 비교"하는 방식으로 학습된다는 점과도 잘 맞는다.

목적함수는 PPO의 클리핑을 그대로 쓰되, 그룹 평균을 취하고 KL 항을 보상이 아니라 **손실에 직접** 넣는다.

$$
\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min\left(\rho_{i,t}\hat A_{i,t},\, \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_{i,t}\right) - \beta\, \mathbb{D}_{KL}[\pi_\theta \,\|\, \pi_\text{ref}]\right\}\right]
$$

여기서 $\rho_{i,t}$는 토큰 단위 확률비이고, KL은 항상 0 이상인 다음 불편 추정량으로 계산한다.

$$
\mathbb{D}_{KL}[\pi_\theta \,\|\, \pi_\text{ref}] = \frac{\pi_\text{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log\frac{\pi_\text{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

풀이 단계마다 보상을 주는 process supervision 버전도 제시한다. 이 경우 각 토큰의 advantage는 그 뒤에 오는 단계들의 정규화된 보상을 더한 값이다. 학습 중 정책이 좋아지면 보상 모델을 다시 학습시키는 **반복 GRPO**도 실험한다.

### SFT, RFT, DPO, PPO를 한 틀로 보기

논문은 여러 학습 방법을 "데이터를 어디서 가져오는가", "보상을 어떻게 주는가", 그리고 토큰별 **기울기 계수**가 무엇인가로 통일해서 비교한다. 예를 들어 SFT는 모든 정답 토큰에 같은 계수를 주고, RFT는 정답으로 판정된 자기 샘플만 학습하며, GRPO는 보상에 따라 계수의 크기와 부호가 달라진다. 이 관점에서 **현재 정책이 직접 만든 샘플로 학습하는 온라인 방식**과 **보상에 따라 계수를 달리 주는 방식**이 유리하다는 결과를 보인다.

## 실험 결과

- DeepSeekMath-Instruct 7B는 MATH에서 46.8%였고, 여기에 GRPO를 적용한 DeepSeekMath-RL 7B는 도구 없이 **51.7%** 를 기록했다. 당시 오픈 7B~70B 모델 중 가장 높았고, 64개 샘플 다수결(self-consistency)로는 60.9%까지 올라갔다.
- RL 학습에는 GSM8K와 MATH의 사고 사슬 형식 질문만 썼는데도, 학습에 쓰지 않은 다른 수학 벤치마크에서도 성능이 올랐다.
- 분석 결과, RL은 **Maj@K(다수결 정확도)는 올리지만 Pass@K(K개 중 하나라도 맞힐 확률)는 거의 올리지 않았다.** 저자들은 RL이 모델에 새로운 능력을 만들기보다, 이미 Top-K 안에 있던 정답이 더 자주 나오도록 출력 분포를 다듬는 쪽에 가깝다고 해석한다.

## 느낀 점

- PPO에서 가장 무거운 부분인 가치 함수를 **"같은 질문에 대한 여러 답의 평균"** 이라는 단순한 기준선으로 대체한 아이디어가 인상 깊었다. LLM은 같은 프롬프트로 여러 번 샘플링하기 쉬우니, 강화학습의 고전적인 기준선(baseline) 아이디어를 LLM 환경에 맞게 다시 쓴 셈이다.
- KL을 보상에 섞지 않고 손실에 직접 넣은 것도 실용적인 선택 같았다. advantage 계산이 그룹 정규화만으로 깔끔하게 유지된다.
- "RL은 Pass@K를 거의 올리지 못한다"는 분석은, 강화학습이 무엇을 할 수 있고 무엇을 못 하는지에 대한 중요한 질문을 던진다. 이 질문은 RL만으로 긴 추론 능력이 생겨나는 것을 보인 다음 글 [DeepSeek-R1](/blog/2026/deepseek-r1/)과 함께 보면 더 흥미롭다.
