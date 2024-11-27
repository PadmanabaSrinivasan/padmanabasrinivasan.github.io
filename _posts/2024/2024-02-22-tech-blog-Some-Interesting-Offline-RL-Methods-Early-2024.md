---
layout: post
title: Some Interesting Offline RL Methods (Early 2024)
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/path.jpg
share-img: /assets/img/path.jpg
tags: [tech, machine learning, reinforcement learning]
---

# Intro

Reinforcement Learning (RL) combined with deep learning has shown incredible promise in recent years: in 2016, AlphaGo was good enough to beat the best Go players; by 2017 AlphaGo Zero learned to play against itself and blew AlphaGo's performance out of the water; and in 2019 MuZero mastered not only Go, but three other games as well while learning the rules of each game itself. As impressive as these algorithms are, each has a voracious appetite for data. AlphaGo is hungry for human-generated data, and its successors generate their own data in extensive trial-and-error style games. In short, deep RL for many real-world problems tends to be data hungry, either for pre-collected data or for data directly collected from interaction. 

Offline RL algorithms learn from pre-collected, fixed datasets that provide poor coverage of the state-action space. Furthermore, the dataset may consist of trajecotories produced by policies of varying quality. Faced with limited data coverage, preventing the policy from selecting out-of-distribution (OOD) actions is the goal of offline RL algorithims. 

In this post, I look at interesting and novel approaches taken by some recent model-free offline RL algorithms.


### 1. Give the Policy an OOD Budget

**[Budgeting Counterfactual for Offline RL](https://arxiv.org/abs/2307.06328)**

**TL;DR** &nbsp; This algorithm tells the policy how much *freedom* it has to select OOD actions via a budget parameter. If there is insufficient budget to perform RL, the algorithm falls back to the behavior policy. 

This paper presents [BCOL](https://arxiv.org/abs/2307.06328) which introduces a budgeting counterfactual, $$b \geq 0$$, to explicitly tell the policy how many deviations from the behavior policy are permissible. Let $$\pi$$ and $$\pi_{\beta}$$ be the current policy and behavior policy, respectively. At each time step $$t$$s, if the policy selects an OOD action the budget is decremented: $$b_{t+1} = b_{t} - \boldsymbol{\texttt{1}} (\pi(\cdot \mid s) \neq \pi_{\beta}(\cdot \mid s)) $$ with an initial budget $$b_0 = B$$, where $$\boldsymbol{\texttt{1}}$$ is the indicator function.

BCOL facilitates value learning using the budget by modifying the Bellman backup operator to:
$$
    \mathcal{T}_{\text{CB}} Q(s, a, b) = r(s, a) + 
    \begin{cases}
        \text{max}( Q(s', \pi(s', b), b-1), Q(s', a', b) ),\quad \text{if}\ b > 0 
        \\
        Q(s', a', b),\quad \text{if}\ b = 0,
    \end{cases}
$$

which falls back to the SARSA update when there is no further budget to perform RL.

BCOL also needs to ensure that for $$b > 0$$, the Q-values satisfy $$Q(s, a, b) > Q(s, a, b-1)$$ which requires additional [CQL-style](https://arxiv.org/abs/2006.04779) budget regularization term to ensure action-values decrease with budget. 

The sampling procedure at inference time also needs to be modified to fall back to the behavior policy when $$b = 0$$:
$$
    \texttt{select}(\pi, \pi_{\beta} ; s, b, Q) =
    \begin{cases}
        \pi_{\beta} (\cdot | s),\quad \text{if}\ Q(\pi(s), s, b-1) \leq Q(\pi_{\beta} (s), s, b)\ \text{or}\ b = 0 
        \\
        \pi (\cdot | s),\quad \text{otherwise},
    \end{cases}
$$

which requires learning an empirical behavior policy $$\pi_{\beta}$$. 

**The Good** &nbsp; The proposed BCOL algorithm proposes an intuitive approach to controlling and curtailing the degree of OOD-ness of the policy and OOD-value overestimation. The fundamental algorithm's flexibility is evidenced by its insertion in both SAC and TD3, with both versions achieving reasonable results. 

**The Bad** &nbsp; I have identified the following limitations of BCOL:

1. **Computational:** BCOL requires an estimate of $$\pi_{\beta}$$, budget regularization and additional forward passes through the Q-functions during both training and inference compared to SAC/TD3 and other offline RL algorithms.
2. **Budget:** The authors provide no insight into how to set the initial budget parameter $$b_{0} = B$$ other than via empirical evaluation in the environment. 
3. **Multi-modal behavior policy:** BCOL fails to tackle the case where the dataset is produced by multiple behavior policies. Their use of a unimodal Gaussian policy $$\pi_{\beta}$$ trained using MLE (forward KL minimization) will fail on heteroskedatic datasets. This is a limitation of many other prior methods that rely on density behavioral policy. For BCOL a parametrised mixture, such as an [MDN](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj80IyJ0vCDAxWgbEEAHVJRAL8QFnoECBgQAQ&url=https%3A%2F%2Fpublications.aston.ac.uk%2F373%2F1%2FNCRG_94_004.pdf&usg=AOvVaw0GQUlGuxl999dBOx0zWpOn&opi=89978449) is easily pluggable into both the operator $$\mathcal{T}_{\text{CB}}$$ and the function $$\texttt{select}$$, though at increased complexity and computational cost. 

**Conclusion** &nbsp; Overall, BCOL's budget parameter sounds interesting on paper, but the authors' application is inefficient compared to the much simpler and faster [IQL](https://arxiv.org/abs/2110.06169). The algorithm exposes multiple points of failure in its need for an empirical behavior policy, critic regularization and substantial hyperparameter tuning. 



### 2. Plan Ahead and Execute

**[Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning](https://arxiv.org/abs/2206.04384)**

**TL;DR** &nbsp; This algorithm exploits the fact that offline RL datasets are relatively small to construct a graph-based transition model and searches for an optimal sequence of actions to execute in the environment.

[VMG](https://arxiv.org/abs/2206.04384) learns a graph structure of the offline dataset using an encoder, $$f_{s, a}$$, to map a state-action tuple into a metric space with the same embedding as the next (reachable) state produced by another encoder $$f_{s'}$$. Given a tuple $$(s, a, s')$$, the *reachability* of $$s'$$ when taking action $$a$$ from $$s$$ is determined by how large the $$L_2$$ distance between the embeddings produced by the two encoders is.

Once transitions have been encoded into a metric space, state embeddings less than a threshold $$\gamma_{m}$$ apart are mapped to the same vertex to form a set of vertices $$\mathcal{V}$$. For two different vertices $$v_i$$ and $$v_j$$, there exists a directed edge $$e_{ij}$$ between the two if there is a transition in the dataset between any state in $$v_i$$ to any state in $$v_j$$. Each directed edge can be assigned a reward, computed by averaging over the state transition rewards of states in the vertices. 

Finally, a dynamics that produces the action $$a = d(s, s')$$ that causes the state transition $$s \rightarrow s'$$, is trained. When rolling out VMG, given an initial state $$s_0$$, a Dijkstra's algorithm is used to search for high-reward paths for $$N_s$$ steps and the path that leads to the graph with the maximum reward is executed. 

**The Good** &nbsp; VMG presents an alternative to the offline methods built on actor-critic algorithms. Consolidating similar states into one vertex in metric space should allow effective "stitching" of sub-trajectories without the pitfalls of off-policy evaluation or the instability of actor-critic training. 

**The Bad** &nbsp; The algorithm requires several hyperparameters that are not easy to select. The threshold $$\gamma_m$$ is controls vertex grouping in the metric space, but itself may depend on the size of the metric space and the nature of the features contained within the observations. The number of steps to search, $$N_s$$, is highly dependent on dataset: for the Antmaze tasks, trajectories can be anything from a few dozen to hundreds of steps long, depending on the maze size. The authors do not indicate how large the search parameter $$N_s$$ should be for episodic Antmaze tasks or how this could be adapted to the continuous Gym Locomotion tasks. In summary, the many hyperparameters of VMG mean than tuning per-dataset is both necessary and tedious. Furthermore, VMG's performance on Locomotion tasks is poor on the $$\texttt{-medium-replay}$$ and $$\texttt{-medium-expert}$$ datasets, suggesting that this graph-based approach fails to generalize to mixed-policy datasets. Finally, the authors do not discuss the inference cost/time when rolling out a policy compared to standard actor-critic algorithms. Given the multi-step search needed after executing each action, VMG is likely far slower. 

**Conclusion** &nbsp; VMG is a compelling alternative to standard offline RL methods on specific kinds of tasks. Hyperparameter optimization remains a challenge, requiring the tuning of five separate algorithmic parameters. VMG remains a competitive algorithm for small datasets, though for larger datasets, actor-critic offline RL may be a better choice.



### 3. Enforce Q-Ensemble Independence

**[SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning](https://arxiv.org/abs/2401.03137)**

**TL;DR** &nbsp; This algorithm identifies why methods with large Q-ensembles fail and proposes a novel regularization loss that encourages independence between ensemble Q-functions that share targets.

The aptly titled [SPQR](https://arxiv.org/abs/2401.03137) is a regimented approach to deploying ensembles while avoiding [diversity collapse](https://arxiv.org/abs/2201.13357). SPQR encourages *independence* between ensemble members that share backup targets and can be applied to both online and offline settings. Offline SPQR achieves some of the best performance of any ensemble-based method, with the exception of [MSG](https://arxiv.org/abs/2205.13703) and [RORL](https://arxiv.org/abs/2206.02829), that I have seen -- though unfortunately, the authors' evaluation on the challenging Antmaze tasks is limited to the $$\texttt{-umaze}$$ and $$\texttt{-medium-play}$$ datasets.

Most methods that use ensembles *assume* increased diversity. The diversity of ensembles is defined through the bias: let $$Q^{*} (s,a)$$ be the optimal value function, for an ensemble of Q-networks, the bias of each network $$e^{i} = Q^{*} (s, a) - Q^{i} (s, a)$$ is assumed to follow a uniform distribution $$e^{i} \sim \mathcal{U} (-\tau, \tau)$$ with mean zero. SQPR's authors show empirically that for large ensembles, the Q-networks show substantial correlation and posit that random initialization is insufficient to maintain diversity. 

The authors look to [spiked models](https://www.jstor.org/stable/2674106), which address the spectra of real-world data that consist of both informative signals and noise. When the entries of a random matrix are sampled from a uniform distribution, the density of the eigenvalues obey [Wigner's semicircle law](https://mathworld.wolfram.com/WignersSemicircleLaw.html) with one single large eigenvalue. For an ensemble of Q functions to be independent, the spectra of random, mean-adjusted action-value samples must follow Wigner's semicircle distribution. The SQPR method minimizes the KL divergence between the ensemble's spectral distribution and a Wigner's semicircle distribution as an auxiliary loss for the value networks. 

The authors evaluate SPQR against min-clipped SAC (large ensemble + no diversification/regularization) and [EDAC](https://arxiv.org/abs/2110.01548) (large ensemble + gradient diversification) and demonstrate that SPQR regularization increases Q function independence translating to enhanced performance in both online and offline settings. 

**The Good** &nbsp; SPQR is a well motivated and general method to realize the benefits of ensembles. The SPQR loss can be plugged in to existing algorithms and improve performance, even for small ensemble methods. The authors also report a minimal increase in training time compared to non-SPQR regularized methods.

**The Bad** &nbsp; In the offline regime, the paper's analysis focuses on Gym Locomotion tasks; these are already well solved by existing algorithms and using SPQR-augmented ensembles yields little additional performance over other ensemble-based methods like EDAC. In Antmaze tasks, the authors switch to augmenting CQL with SPQR and demonstrate that performance improves with just two critics -- ensemble methods struggle on these tasks and a more informative experiment would test whether an SQPR-enhanced ensemble performs comparably to [MSG]((https://arxiv.org/abs/2205.13703)) or [RORL](https://arxiv.org/abs/2206.02829). As previously mentioned, the authors do not provide results for the Antmaze $$\texttt{-large}$$ datasets.  

**Conclusion** &nbsp; SQPR is an interesting take on enforcing and ensuring that models in an ensemble do not collapse. Experiments show that SPQR performs at least as well as other ensemble-based methods and may yield benefits when used alongside existing algorithms. An interesting line of investigation in offline RL would be to see how SQPR affects performance in policy-regularized methods -- I expect all policy regularized methods to benefit from SQPR.


# Finishing Up

The three algorithms discussed in this post pose novel and non-standard approaches to offline RL. Though their performance on standard benchmarks is hardly SOTA, I believe each method has merit and weaknesses can be addressed to yield better performing algorithms.