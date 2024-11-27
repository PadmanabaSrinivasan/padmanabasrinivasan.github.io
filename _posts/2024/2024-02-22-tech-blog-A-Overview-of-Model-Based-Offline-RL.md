---
layout: post
title: An Overview of Model-Based Offline RL Methods
subtitle: 
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/path.jpg
share-img: /assets/img/path.jpg
tags: [tech, machine learning, reinforcement learning]
---

# Intro

### Model-Free vs Model-Based RL

In Reinforcement Learning (RL) tasks, an *agent* interacts with an environment by executing an action, *a*, given the current state, *s* and transitions to the next state *s'* defined by the environment. The environment yields a scalar reward *r* at transition which, with successive interaction, can be used as a signal to learn which actions yield the most reward over many steps. 

The agent aims to learn an *optimal policy* that maximizes the reward it receives over the long run -- any policy can be considered a strategy that the agent executes and the optimal policy lies in the subset of all policies as a policy that maximizes the expected return.

In RL, the decision making problem is often formulated as a Markov Decision Process (MDP) that represents the dynamics of an environment. An MDP consists of a transition function, $$\mathcal{T}$$, which models which state the agent will move after it executes an action, and a reward function, $$\mathcal{R}$$, that issues a reward for the change. Collectively, the transition and reward functions form a model of the environment. 

*Model-free* methods learn optimal policies by directly interacting with environment, collecting experience and exploring the environment to *reinforce* its beliefs about the environment. With repeated environment interaction and improvement, an optimal policy may be learned. 

A drawback of model-free methods is their need to interact directly with the environment to learn in a trial-and-error manner; this can be expensive and take a a long time for a good policy to be learned. *Model-based* algorithms attempt to speed up the learning process by: $$1)$$ estimating the environment model and $$2)$$ rolling out the policy in the estimated environment and augmenting learning with synthetic data. Model-based RL (MBRL) methods are oftne built on top of existing online RL methods with several approaches such as [MBPO](https://arxiv.org/abs/1906.08253) built on SAC, [PlaNet](https://arxiv.org/abs/1811.04551) as well as DQN and [Dreamer](https://arxiv.org/abs/1912.01603) variants, to name a few. 


### Model-Based Offline RL

Offline RL aims to learn optimal policies from a static, fixed datasets with no access to the environment for learning/optimization and model-based offline RL simply applies a trained dynamics model to augment offline data with synthetic rollouts. I have found few survey papers addressing offline MBRL, [this](https://arxiv.org/abs/2305.03360) survey provides a reasonable starting point with summaries of a small selection of papers and [this](https://american-cse.org/csci2022-ieee/pdfs/CSCI2022-2lPzsUSRQukMlxf8K2x89I/202800a315/202800a315.pdf) a comparitive study betweeen model-free and model-based offline RL. This blog post aims to discuss a wider and more recent variety of offline MBRL methods and group them into categories based on offline constraint paradigm. 

The primary challenge facing offline RL is as follows: the static dataset provided to offline RL algorithms has poor state-action space coverage and the neural net function approximators used to learn the value function are prone to ovestimating OOD action-values -- a policy that selects OOD actions will likely perform poorly in the real environment. Model-free offline RL algorithms typically aim to force the policy to select in-distribution actions either by regularizing the critic, or by imposing constraints on the actor. Offline MBRL has similar goals with the added flexibility (and challenge!) of augmenting training with synthetic data -- this can enable more effective learning when substantial "stitching" of sub-trajectories is needed to learn an optimal policy. 

I classify offline into one of two approaches: $$1)$$ dynamics-aware policy constraint (DAPC) and $$2)$$ ensemble-based critic regularization. The former methods are fewer in number and subsets, hence I address those first. 


##### Dynamics-Aware Policy Constraint 

DAPC methods deviate substantially from the typical [dyna-style](https://dl.acm.org/doi/10.1145/122344.122377) MBRL as they do not use a dynamics model for synthetic policy rollouts. I am aware of two methods that fall under this category. 

###### [State Deviation Correction](https://ojs.aaai.org/index.php/AAAI/article/view/20886)

The first DAPC method I know if is [SDC](https://ojs.aaai.org/index.php/AAAI/article/view/20886) which recognizes the limited coverage of the state-action space and designs a method to encourage the policy to move towards observed states from unknown ones; given a dataset tuple $$\{s, a, r, s'\}$$, they construct a perturbed state:
 
$$\hat{s} = \texttt{perturb}(s) = s + \beta \epsilon$$ 
 
with Gaussian noise $$\epsilon \sim \mathcal{N}(0, 1)$$ and standard deviation scale $$\beta$$ and train a policy that will aim to move to $$s'$$ from both $$s$$ and $$\hat{s}$$. SDC trains a standard, [MOPO-style](https://arxiv.org/abs/2005.13239) dynamics ensemble and a CVAE state-transition model that learns all transitions $$s \rightarrow s'$$ in the dataset. 

When training the policy, SDC samples a possible next state from the CVAE given the state $$U(s) = s'$$, constructs a perturbed tuple using $$\texttt{perturb}(\cdot)$$ to produce $$\{\hat{s}, a, r, s'\}$$ and samples the next state from the dynamics model $$D(\hat{s}, \pi(\hat{s}))$$ and uses the policy objective:

$$
    max_{\pi} Q(s, \pi(s)) \quad\text{s.t}\quad \alpha d(U(s' \mid s) \mid\mid D(\hat{s}, \pi(\hat{s}))) \leq \eta
$$

where $$\alpha > 0$$ becomes a Lagrangian multiplier, $$\eta$$ is a divergence threshold and $$d(\cdot, \cdot)$$ is a divergence meausure, for which the authors use [MMD](https://arxiv.org/abs/0805.2368). 

**Overall** &nbsp; SDC training trains a policy that is consistent with dataset-dynamics; regularizing using the perturbed state $$\hat{s}$$ allows for some small exploration around dataset states controlled by the parameter $$\beta$$. The method is reminiscent of policy constraints in model-free offline RL, though the authors demonstrate more robust performance in datasets contructed with a large degree of mixed suboptimality. SDC is highly sensitive to the value of $$\beta$$ and $$\eta$$, and $$\alpha$$ is tuned via dual gradient descent which adds to sources of potential instability. My main criticism of the work is the need to train ensemble dynamics but then never use them for rollouts! 

###### [Recovering from Out-of-sample States via Inverse Dynamics](https://neurips.cc/virtual/2023/poster/72844)

This paper presents [OSR](https://neurips.cc/virtual/2023/poster/72844) with a premise and goal identical to that of SDC; OSR aims to guide the policy to reach dataset-known states when faced with unknown ones. To this end, OSR learns an inverse dynamics model that estimates the action needed to transition between two states. This inverse dynamics model is characterized as a VAE that predicts a Gaussian action that induces the transition between two consective states in the dataset: $$I(a \mid s, s')$$. When the policy is presented with a perturbed state (as with SDC), it is trained to act in order to reach the next known (un-perturbed) state:

$$
    \max_{\pi} Q(\hat{s}, \pi(\hat{s})) - \lambda D_{\text{KL}} (I(a \mid \hat{s}, s') \mid\mid \pi(a \mid \hat{s}))
$$

where $$\lambda > 0$$ is the constraint strength and constraint optimizes the KL-divergence between two Gaussians. This is a simple policy constraint with a dynamics-aware explicit density model; the authors also propose a critic regularization alternative which applies CQL-style regularization where action-values sampled from the policy are penalized while those from the inverse dynamics model are favored.

**Overall** &nbsp; OSR's results are impressive and it seems like a far less resource-wasteful algorithm than SDC. Using the VAE to produce Gaussian actions followed by a KL constraint is a clean way of overcoming the density estimation and [forward/reverse KL divergence problem](https://dibyaghosh.com/blog/probability/kldivergence.html) as now, both distributions are Gaussian. Estimating the behavioral policy using a VAE, while capable of producing explicit densities, is limiting as VAEs are known to find it [challenging to learn complex modes](https://openreview.net/forum?id=5Spjp0zDYt), especially compared to implicit density estimation method. OSR retains the sensitivity to state perturbation scalar $$\beta$$. 

**Concerns** &nbsp; I do not understand how OSR is able to perform as well as reported, while effectively being constrained to a SAC+BC-style constraint: assuming a deterministic transition function, the policy is always constrained towards the action $$a$$ that leads to the transition $$\hat{s} \rightarrow s'$$ where $$s'$$ is from the dataset and $$\hat{s}$$ is the perturbed state that is nearly identical to $$s$$, according to Theorem 2. As the next state is never estimated by a dynamics model, the inverse dynamics model always produces the exact behavioral action that leads to the tuple-defined transition and the subsequent KL constraint means that this is (nearly) equivalent to SAC+BC and for single policy datasets, this should be identical to learning an explicit empirical behavioral policy using BC and plugging that into the KL constraint in place of the inverse dynamics model. I don't understand *how* reported performance can be so good as the resulting policy and value function should be closer to that learned with SARSA. 


###### [Policy Constraint with Offline MBRL](https://arxiv.org/abs/2206.07166)

This paper introduces [SDM-GAN](https://arxiv.org/abs/2206.07166) which adpots a typically model-free (style) of offline constraint. What's more, unlike the previous two policy-constraint methods, this one trains any uses ensemble dynamics to generate synthetic training data. 

As the name suggests, the authors train a discriminator using the dataset's state-action tuples as the *real* samples and combines samples policy actions for states in the dataset and for synthetically generated states as the *fake* samples. The authors prove that training a policy (generator) to maximize reward and fool the discriminator is equivalent to an objective that matches the unnormalized occupancy distributions between the behavioral policy and the current policy. 

**Overall** &nbsp; SDM-GAN's theoretical results are convincing, but the empirical evaluation yields scores that are middling at best. The method also seems **very** tricky to tune both in a hyperparameter value sense and the fact that learning dynamics mean that the policy is trying to optimize two adversarial objectives (Q function + state-action discriminator) at the same time. Finally, looking at the high level view of the approach, it seems pretty similar to [COMBO](https://arxiv.org/abs/2102.08363) (I cover COMBO later on) except SDM-GAN seperates the discriminate and value-learning functions.

**Concerns** &nbsp; The authors make no attempt to ensure the validity of synthetically generated states (i.e. no uncertainty penalties or episode truncation) which may be another factor contributing to subpar performance. Though the discriminator will eventually be trained to reject these transitions, the problem of value overstimation propagating through remains. The authors use the vanilla GAN objective which minimizes JS-divergence; they also experiment with the Wasserstein-1 distance (WGAN) which should perform better, though in practice they show this performs substantially worse.


##### Critic Regularized Offline MBRL

Critic regularization methods in offline MBRL display much more variety. Generally, I classify these methods into of two kinds: 

1. Uncertainty penalty methods: these estimate some version of uncertainty using dynamics ensembles and apply a value penalty that incorporates the estimate.
2. Adversarial value minimization methods: these tend to assume that *any* state-action samples sampled from dynamics are OOD and apply a [CQL-inspired](https://arxiv.org/abs/2006.04779) auxiliary objective for conservative value estimation.

##### MoREL

[MoREL](https://arxiv.org/abs/2005.05951) is one of the earliest approached to offline MBRL that proposes a simple and intuitive extension to the online [MBPO]((https://arxiv.org/abs/1906.08253)). MoREL recognizes that a trained ensemble of dynamics models will 'agree' on transition dynamics when the transition is observed in the offline dataset, and will 'disagree' otherwise. In practice, the notion of agreement is not a hard boundary: when transitions are similar to the ones in the dataset, a measure of disagreement between predictions of the ensemble dynamics should be lower than for transitions unobserved. 

Let $$disc(s, a)$$ denote a function that computes the discrepancy in the next state between dynamics models, MoREL designs an unknown state-action detector (USAD) that simply detects if disrepancy is above a threshold $$\alpha$$:

$$
    USAD(s, a, \alpha) = 
    \begin{cases}
        \text{True},\quad disc(s, a) > \alpha 
        \\
        \text{False},\quad \text{otherwise}.
    \end{cases}
$$

The USAD detector allows detection of when a rollout has potentially gone OOD: if the USAD detector return True, then a rollout has likely gone into an OOD region and the rollout can be halted and the reward assigned to this particular state-action pair can be a large negative penalty; this means any synthetic data generated using the dynamics models form a pessimistic MDP (p-MDP) that punishes policies that visit unknown states, where the halting operation is a property unique to the p-MDP. Key design factors now become the choice of $$disc(s,a )$$, the value $$\alpha$$ and the halt penalty $$-\kappa$$. For the former, the authors choose the maximum L2-norm between the next state predictions of any pair of dynamics models in the ensemble, and the remainder of the parameters are tuned on a per-dataset basis. 

**Overall** &nbsp; MoREL's approach is simple and intuitive, directly leveraging ensemble-based uncertainty detection to prevent OOD value overstimation from affecting learning. Detecting whether a rollout is moving into unknown regions is a technique that (to the best of my knowledge) no other methods use and it seems as though this makes for effective constraining. 

**Concerns** &nbsp; While MoREL performs well among offline MBRL methods and early model-free methods, its performance lags behind simpler, contemporary model-free methods such as IQL and TD3+BC. The best part of this paper is the development of the p-MDP as an approach for future work. 


##### MOPO

Roughly contemporaneous with MoREL, [MOPO](https://arxiv.org/abs/2005.13239) makes use of the dynamics ensemble in an entirely different way. Rather than detecting uncertainty and penalizing the truncated state reward, MOPO estimates uncertainty using the ensemble at each step and augments the reward estimate with an uncertainty penalty:

$$
    r(s, a) = \hat{r}(s, a) - \lambda u(s, a)
    \\
    \hat{r} (s, a) = \frac{1}{N} \sum_{i=1}^{N} r_{i} (s, a)
    \\
    u(s, a) = \text{max}_{i = 1}^{N} \lvert\lvert \Sigma_i(s, a) \rvert\rvert_{F},
$$

where $$\hat{r}(s, a)$$ is the average reward predicted by $$N$$ dynamics models and the uncertainty is the maximum of the norms of the standard deviations predicted by the dynamics models.

MOPO allows the policy to (potentially) visit unkown states, but compensates for out-of-support states using pessimism in the form of a lower confidence bound (LCB)

**Overall** &nbsp; MOPO makes effective, though simple, use of dynamics models to generate synthetic data and perform a LCB update. As an early offline RL method, MOPO is landmark for both offline MBRL and performs well against early model-free methods.

**Concerns** &nbsp; MOPO's performance is poor compared to even the (relatively) old CQL, which is arguably far simpler to implement -- though MOPO's fundamental practice of using the ensemble for uncertainty estimation has been adopted by several newer methods. 


##### COMBO

It is at this point that offline MBRL starts to get interesting. [COMBO](https://arxiv.org/abs/2102.08363), fundamentally proceeds similarly to CQL: to recap, CQL adversarailly assumes that any actions sampled from the policy are OOD and subsequently, applies a penalty that raises the value of in-distributions actions and penalizes those of OOD actions. COMBO makes a broader assumption that any *states* from the the ensemble dynamics models and any *actions* from the policy on these states are all OOD and are penalized, while those dataset state-action tuples are not. This results in a change in paradigm where the policy is steered towards known, optimal states. 

**Overall** &nbsp; COMBO is the probably the most interesting algorithm so far both in terms of its analysis and in *what exactly* the algorithm aims to achieve -- allowing synthetic data augmentation while guiding the policy to fllow in-dataset trajectories.  While COMBO's results in D4RL are hardly exceptional, experiments on image-based environments (walker-walk and swayer-door) suggest that COMBO-train policies generalize well. 

**Concerns** &nbsp; My only criticism of this algorithm is the computational complexity -- CQL is well known to be a slow algorithm to train and COMBO with additional model-based dynamics is going to be even worse!
