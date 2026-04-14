# Safe Critic：安全评价函数

> 本文介绍安全强化学习中安全评价函数（Safe Critic）的设计思路，即如何训练一个专门评估动作安全性的价值函数。

## 1. 背景与动机

在约束马尔可夫决策过程（CMDP）框架下，智能体需要同时优化两个目标：
- **任务目标**：最大化累积奖励 $J_R(\pi) = \mathbb{E}[\sum_t \gamma^t r_t]$
- **安全约束**：满足累积代价约束 $J_C(\pi) = \mathbb{E}[\sum_t \gamma^t c_t] \leq d$

其中 $c_t$ 是代价函数（cost function），当智能体进入不安全状态或执行危险动作时 $c_t > 0$，$d$ 是允许的代价上限。

标准的 Actor-Critic 方法只训练一个 Critic 来估计奖励的价值函数 $V_R(s)$。在 SafeRL 中，我们额外需要一个 Safe Critic 来估计代价的价值函数 $V_C(s)$，用于判断当前策略是否满足安全约束。

## 2. Safe Critic 的基本形式

### 2.1 代价价值函数 (Cost Value Function)

$$V_C^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t c(s_t, a_t) \mid s_0 = s\right]$$

$V_C^\pi(s)$ 估计从状态 $s$ 出发，遵循策略 $\pi$ 的期望累积代价。如果 $V_C^\pi(s_0) > d$，说明当前策略违反了安全约束。

### 2.2 代价 Q 函数 (Cost Q-Function)

$$Q_C^\pi(s, a) = c(s, a) + \gamma \mathbb{E}_{s' \sim P}[V_C^\pi(s')]$$

$Q_C^\pi(s, a)$ 估计在状态 $s$ 执行动作 $a$ 后的期望累积代价，可以用于在动作层面评估安全性。

### 2.3 训练方式

Safe Critic 的训练与标准 Critic 类似，使用 TD 学习：

$$\mathcal{L}(\phi) = \mathbb{E}\left[(V_C^\phi(s_t) - (c_t + \gamma V_C^\phi(s_{t+1})))^2\right]$$

## 3. Safe Critic 在策略优化中的应用

### 3.1 拉格朗日方法

最常见的做法是将 CMDP 转化为拉格朗日对偶问题：

$$\min_\lambda \max_\pi J_R(\pi) - \lambda (J_C(\pi) - d)$$

Safe Critic 提供 $J_C(\pi)$ 的估计，拉格朗日乘子 $\lambda$ 根据约束违反程度自动调整：
- 如果 $V_C > d$（违反约束），增大 $\lambda$，加大安全惩罚
- 如果 $V_C < d$（满足约束），减小 $\lambda$，放松安全限制

### 3.2 安全层 / 安全投影

另一种思路是在策略输出后添加一个安全层，将不安全的动作投影到安全集合中：

$$a_{\text{safe}} = \arg\min_a \|a - a_\pi\|^2 \quad \text{s.t.} \quad Q_C(s, a) \leq d'$$

Safe Critic 在这里提供约束函数 $Q_C(s, a)$ 的估计。

### 3.3 CPO (Constrained Policy Optimization)

CPO 在 TRPO 的基础上引入安全约束，使用 Safe Critic 的估计来线性化约束：

$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_{s \sim d^{\pi_k}}[\nabla_\theta J_R \cdot (\theta - \theta_k)]$$
$$\text{s.t.} \quad J_C(\pi_k) + \nabla_\theta J_C \cdot (\theta - \theta_k) \leq d$$

## 4. Safe Critic 的挑战

- **稀疏代价信号**：在许多环境中，不安全事件很少发生，导致 $c_t$ 大部分时间为 0，Safe Critic 难以学到有效的安全估计。
- **分布偏移**：Safe Critic 在当前策略的数据上训练，但需要评估更新后策略的安全性，存在分布偏移问题。
- **保守估计 vs 准确估计**：为了安全，通常希望 Safe Critic 给出偏保守的估计（高估代价），但过于保守会限制策略的探索能力。

## 参考资料

- Achiam et al. "Constrained Policy Optimization." ICML, 2017.
- Ray et al. "Benchmarking Safe Exploration in Deep Reinforcement Learning." arXiv, 2019.
- Stooke et al. "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML, 2020.
