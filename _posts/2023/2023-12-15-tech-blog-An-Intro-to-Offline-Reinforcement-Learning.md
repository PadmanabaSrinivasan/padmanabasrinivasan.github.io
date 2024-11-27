---
layout: post
title: An Intro to Offline Reinforcement Learning
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/path.jpg
share-img: /assets/img/path.jpg
tags: [tech, machine learning, reinforcement learning]
---

## What is Reinforcement Learning?

Say you have bought a table from a certain Swedish retailer famous for both their fairly priced furnite, and meatballs, but have lost the instructions to assemble it. A trivial, albeit cumbersome approach would be to just try putting different parts together and to make it look more and more like the image on the box. Through repeated attempts, you learn about how different parts fit together until mastery, at which point you are rewarded with a completed table. 

This process of repetition and feedback describes the fundamental process of Reinforcement Learning (RL). More abstractly, in RL we aim to train an agent to maximize a reward by interacting with the environment. This is usually achieved by allowing the agent to take actions and receive a reward as feedback and repeating this process until it learns an *optimal policy* to solve the task. 

Rephrasing all this more formally, the agent follows a policy $$\pi \in \mathcal{\Pi}$$ in an **environment** by selecting an action $$a \in \mathcal{A}$$ based on the current state $$s \in \mathcal{S}$$. Once an action is taken, the environment produces a **reward** $$r \in \mathcal{R}$$ which is used as feedback to the agent. By executing an action, the agent is able to effect a change in the environment that causes a transition to a new state $$s' \in \mathcal{S}$$. The way in which environment dynamics work is defined by a **model**, which may be known or approximated allowing the use of model-based RL algorithms, or unknown in which case model-free RL algorithms must be used.

The agent's goal is to learn a policy $$\pi(s)$$ that maximizes the expected future reward and is taught to do so using a value function $$V(s)$$ that predicts the expected future reward from that state following the policy $$\pi$$.


## From RL to Offline RL

The agent needs to be trained on data collected by interacting with the environment over a series of time steps $$1, 2, ..., T$$ expressed as a trajectory: $$S_1, A_1, r_1, S_2, A_2, r_2, ..., S_T, A_T, r_T$$. **On-Policy** RL generates trajectories using the current policy $$\pi_k$$ and learns from this to produce the next policy $$\pi_{k+1}$$ to produce a sequence of policies that successively improve. **Off-policy** RL algorithms generate some trajectories using the current policy and
store trajectories from past policies in a replay buffer. Samples from the replay buffer therefore contain samples from a mixture of past policies and these are sampled to produce the next policy $$\pi_{k+1}$$. The replay buffer can be updated periodically to add new experiences and remove old ones. In the **Offline** RL setting, the alrogithm is further constrained to using a fixed dataset of trajectories collected produced by (potentially unknown) behavioral policies, with no ability to interact with and explore the environment.

In the offline domain, only the values of the actions present in the dataset can be empirically learned. Standard off-policy RL algorithms have no way of evaluating and exploring the how good out-of-distribution (OOD) actions and so, the neural networks used to learn the value will extrapolate in OOD regions and [overestimate their value](https://offline-rl-neurips.github.io/pdf/41.pdf) that the trained policy may select. In reality these actions may be quite poor, so when the policy is deployed in the environment it will execute these untrusted actions and perform poorly. 

Clearly, in order to learn effectively online, an offline RL algorithm must both learn in a similar way to off-policy algorithms while also either: 

1. Directly address the extrapolation problem and actively push down the values of OOD actions;
2. Constrain the actor to select those actions present in the dataset. 


## Conclusion

Standard online deep learning methods have achieved incredible success in recent years, from playing [Go](https://en.wikipedia.org/wiki/AlphaGo) to [conversational agents](https://en.wikipedia.org/wiki/ChatGPT) which learn by interactng with the environment. In many domains though, it may be expensive or impossible to directly interact with the environment such as when training [self-driving cars](https://arxiv.org/abs/2110.07067), [robotic surgeons](https://arxiv.org/abs/2105.01006) (including [safe methods](https://arxiv.org/pdf/2109.02323.pdf) and other [clinical](https://arxiv.org/abs/2002.03478) applications) and [sports modelling](https://www.ijcai.org/proceedings/2020/0464.pdf). Offline RL algorithms can learn from demonstrations produced by humans, that are potentially suboptimal and "stitch" together optimal trajectories from suboptimal data in a way that is [sample efficient](https://arxiv.org/abs/2106.04895). 

I plan to do a more comprehensive [review](https://arxiv.org/abs/2203.01387) of current offline RL methods as there have been many exciting developments in 2023 that have seen notable success; yet many of these methods also exhibit some disappointing trends in offline RL. I will cover some of of the most interesting papers of the year in my next post and my thoughts on each approach. 