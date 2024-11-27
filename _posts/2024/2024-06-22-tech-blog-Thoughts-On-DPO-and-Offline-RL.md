---
layout: post
title: Thoughts on DPO and Offline RL
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/path.jpg
share-img: /assets/img/path.jpg
tags: [tech, machine learning, reinforcement learning]
---


[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) is all the rage now in LLMs, and rightly so! The derivation is neat (and very familiar to those experienced with reinforcement learning) and allows direct, preference-based finetuning of regression-trained LLMs without having to learn a reward model.

In this post, I want to explore what implications a DPO-style training can offer to offline RL. To this end, I begin with the derivation.

Consider the following optimization problem:

$$
    max_{\pi}\quad  \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} [r(s, a)]\quad s.t.\quad D_{\text{KL}} [\pi (a | s) \mid\mid \pi_{\text{ref}} (a | s)] \leq \epsilon,
$$

which maximizes a reward function while constraining the KL--divergence between the current policy and a reference policy. Rewriting the objective using the Lagrangian multiplier $$\beta$$, we get:

$$
    max_{\pi}\quad  \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} [r(s, a)] - \beta D_{\text{KL}} (\pi(a | s) \mid\mid \pi_{\text{ref}} (a | s))
    \\
    = max_{\pi}\quad \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} [r(s, a) - \beta \log \frac{\pi(s | a)}{\pi_{\text{ref}} (s | a)}]
    \\
    = min_{\pi}\quad \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} [\log \frac{\pi(s | a)}{\pi_{\text{ref}} (s | a)} - \frac{1}{\beta} r(s, a)]
    \\
    = min_{\pi}\quad \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} [ \log \frac{\pi(a | s)}{\pi_{\text{ref}} (a | s) \exp (\frac{1}{\beta} r(s, a)) \frac{1}{Z(s)}}  - \log Z(s)],
$$

where 

$$
    Z(s) = \int_{\mathcal{A}} \pi_{\text{ref}} (a | s) \exp(\frac{1}{\beta} r(s, a)),
$$

is the state-dependent partition function.

The (estimated) reward can then be expressed as:

$$
    r(s, a) = \beta \log \frac{\pi(a | s)}{\pi_{\text{ref}} (a | s)} + \beta \log Z(s)
$$

This poses a problem as $$Z(s)$$ requires integrating over actions to compute -- in practice this is intractable for continuous $$\mathcal{A}$$. Most prior methods tends to ignore normalizing constant and attempts to approximate are rendered both [more computationally expensive and less performant](https://arxiv.org/abs/2006.09359). 

The authors of DPO notice that rather than optimizing the objective directly, they can instead optimize the difference:

$$
    r(s, a_1) - r(s, a_2) = (\beta \log \frac{\pi(a_1 | s)}{\pi_{\text{ref}} (a_1 | s)} + \beta \log Z(s)) - (\beta \log \frac{\pi(a_2 | s)}{\pi_{\text{ref}} (a_2 | s)} + \beta \log Z(s))
    \\
    = \beta (\log \frac{\pi(a_1 | s)}{\pi_{\text{ref}} (a_1 | s)} - \log \frac{\pi(a_2 | s)}{\pi_{\text{ref}} (a_2 | s)}),
$$

where the partition functions cancel out, resulting in a direct optimizion of log-probabilities. When training using preference-annotated data, the preference oracle denotes (absolute) paired preferences $$p(a_1 \succ a_2)$$ that can be directly regressed via a [Bradley-Terry preference model](https://en.wikipedia.org/wiki/Bradley–Terry_model) to yield the DPO objective.


**What does this mean for offline RL?** 

First off, DPO is a *finetuning* process that requires the base policy $$\pi_{\text{ref}}$$ to be a pretrained LLM -- this can be a simple MLE-trained model -- that is nudged to generate human-preferred sequences from an offline buffer. At no point is are samples drawn from $$\pi$$ and instead, $$a_1$$ and $$a_2$$ are drawn from $$\pi_{\text{ref}}$$. This is distinctly different from an RLHF-based approach that learns a reward model and requires humans to preference-rank samples drawn from a series of policies. Consequently, DPO is a far simpler and more stable approach to training LLMs. 

Despite the *neatness* of DPO, by limiting the algorithm of offline finetuning based only on samples from a maximum likelihood model $$\pi_{\text{ref}}$$ seems like it might limit the extent of improvement (or, to put it another way, human-preferred alignment) and this could be the case for many subsequent [offline refinement methods](https://arxiv.org/pdf/2402.05749). 

Does DPO have any bearing on offline RL? Simply put, yes: several preference-inspired offline RL methods (for continuous control) have been proposed, such as (non-exhaustive):

1. Inverse-RL-inspired [IPL](https://arxiv.org/abs/2305.15363) which learns a preference-based reward function and subsequently trains an RL policy.
2. [OPAL](https://arxiv.org/abs/2107.09251) trains a policy on high-reward-seeking preferences between subtrajectories (following the behavioral policy).
3. [DPPO](https://arxiv.org/abs/2301.12842) directly uses human-labelled datasets to train human-aligned policies. 
4. [OPPO](https://arxiv.org/abs/2305.16217) and [PT](https://arxiv.org/abs/2303.00957) predict entire  to align with human-preferred trajectories. 
 
A commonality between these methods is their reliance on on-policy trajactories/value-estimation. A key driver of the development of offline RL methods is the ability to enable *stitching* together of suboptimal (behavior-policy-generated) trajectories to improve on the behavior policy. I have yet to come across any preference-inspired methods that can move beyond on the on-policy setting (and is perhaps why I haven't seen much $$\texttt{antmaze}$$ evaluation).

The DPO trick also brings other challenges for offline RL: in practice we want to use a Q function to estimate rewards with which the **deadly triad** can be notoriously unstable and overestimate values for OOD actions. We also need to approximate $$\pi_{\text{ref}}$$ beyond the challenges of the quality of approximation, we do not know whether the behavior policy is multimodal (or how multimodal it is) which often necessitates VAEs (which produces explicit density estimates that can only be extracted by sampling) or MDNs (which have their own, set of training challenges + need to know how multimodal a behavior policy is beforehand). 

Assuming that we have solved the aformentioned challenges, offline RL still poses yet another challenges: multimodal behavior policies can produce multimodal reward functions. DPO (and -like) objectives assume we can sample $$a_1, a_2$$ such that $$a_1 \succ a_2$$, but real-world offline datasets offer no such guarantee of paired-ness. Prior preference-based methods conventiently overcome this challenge by operating at the trajectory level or by using specific, preference-annotated datasets.

The key point I want to make in this, now long-winded post, is that the DPO tricks could enable a very interesting set of approaches to offline RL, provided we can overcome the fundamental challenges of value estimation, density estimation and limitations of on-policy evaluation. 