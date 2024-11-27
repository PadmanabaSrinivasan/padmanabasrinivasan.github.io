---
layout: post
title: An Introduction to Preference-Based RL
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/path.jpg
share-img: /assets/img/path.jpg
tags: [tech, machine learning, reinforcement learning]
---

# Intro

In Reinforcement Learning (RL), an agent navigates through an environment and learns to maximize a discounted sum of scalar rewards produced as a property of the environment. In practice, however, no environment *decides* to provide a reward signal -- such signals are the product of human design. 

At its simplest, a reward might consist of a simple binary signal that indicates when a task has been successfully completed at termination. Such sparse feedback may pose challenges to learning, resulting in the need for [reward shaping](https://www.cdf.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf). More considerate reward design can improve learning efficiency. However, this requires far more effort in design to ensure that the reward signal is informative and robust to [reward hacking](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html) where flaws in the reward signal are exploited by the agent, resulting in undesireable behavior -- see [this](https://openai.com/index/faulty-reward-functions/) for an illustrated example. 

Clearly, for many tasks constructing an informative reward function is challenging because without the ability to explicitly communicate which behaviors are preferred, the agent can exploit the reward function. The discrepancy between human preferences and reward function-implied behavior leads to *misalignment* between the desired and actual behavior of RL-trained policies (see [Bostrom](https://www.goodreads.com/book/show/20527133-superintelligence) and [Russell](https://www.scientificamerican.com/article/should-we-fear-supersmart-robots/#:~:text=The%20machine's%20purpose%20must%20be,way%20it%20sidesteps%20Wiener's%20problem.)).

[Reinforcement Learning from Human Feedback (RLHF)](https://arxiv.org/abs/1706.03741) is one paradigm designed to align RL policy performance with human preferences; the key difference between RLHF and typical RL is in how human preferences are incorporated and iteratively used to refine a policy to produce human-preferred results. 

RLHF underpins many successful applications: first and foremost is to finetune pretrained large language models (LLMs) such as [InstructGPT](https://openai.com/index/instruction-following/) ([this blog](https://huyenchip.com/2023/05/02/rlhf.html) does a great job of describing RLHF in LLMs very well), 