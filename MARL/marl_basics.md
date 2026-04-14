# MARL 基础概念与经典算法

> 本文介绍多智能体强化学习（MARL）的基本概念框架和经典算法，从单智能体 RL 到多智能体的扩展。

## 1. 从单智能体到多智能体

### 1.1 单智能体 RL 回顾

单智能体强化学习的标准框架是马尔可夫决策过程（MDP）：$(S, A, P, R, \gamma)$，其中智能体在状态 $s$ 下选择动作 $a$，环境返回奖励 $r$ 并转移到新状态 $s'$。目标是学习最优策略 $\pi^*$ 最大化累积折扣奖励。

### 1.2 多智能体扩展：随机博弈

当环境中有多个智能体时，MDP 扩展为随机博弈（Stochastic Game）或马尔可夫博弈：

$$(N, S, \{A_i\}_{i=1}^N, P, \{R_i\}_{i=1}^N, \gamma)$$

- $N$：智能体数量
- $A_i$：智能体 $i$ 的动作空间
- $P(s'|s, a_1, ..., a_N)$：联合转移概率，依赖于所有智能体的动作
- $R_i(s, a_1, ..., a_N)$：智能体 $i$ 的奖励函数

核心挑战：每个智能体的最优策略依赖于其他智能体的策略，而其他智能体的策略也在不断变化——环境对每个智能体来说是非平稳的。

### 1.3 问题设定分类

| 设定 | 奖励关系 | 典型场景 |
|------|----------|----------|
| 完全合作 | $R_1 = R_2 = ... = R_N$ | 机器人协作搬运 |
| 完全竞争（零和） | $R_1 = -R_2$ | 围棋、扑克 |
| 混合博弈 | 各自独立 | 自动驾驶交通 |

## 2. 集中式训练分散式执行 (CTDE)

CTDE（Centralized Training with Decentralized Execution）是 MARL 中最主流的训练范式：

- **训练阶段**：可以访问全局信息（所有智能体的观测和动作），用于训练更好的策略
- **执行阶段**：每个智能体只能基于自己的局部观测做决策

这种范式的优势在于：训练时利用全局信息缓解非平稳性问题，执行时保持分散式的可扩展性。

## 3. 经典算法

### 3.1 IQL (Independent Q-Learning)

最简单的方法：每个智能体独立训练自己的 Q 函数，将其他智能体视为环境的一部分。

$$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha [r_i + \gamma \max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i)]$$

**优点**：简单，可扩展性好。**缺点**：忽略了智能体间的交互，环境非平稳导致训练不稳定。

### 3.2 QMIX (Rashid et al., 2018)

QMIX 是值分解方法的代表，核心思想是将全局 Q 值分解为各智能体局部 Q 值的单调组合：

$$Q_{\text{tot}}(s, \mathbf{a}) = f_{\text{mix}}(Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_N(o_N, a_N); s)$$

其中混合网络 $f_{\text{mix}}$ 满足单调性约束：

$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0, \quad \forall i$$

单调性保证了全局最优动作可以通过各智能体独立选择局部最优动作来实现：

$$\arg\max_{\mathbf{a}} Q_{\text{tot}} = (\arg\max_{a_1} Q_1, ..., \arg\max_{a_N} Q_N)$$

**混合网络的实现**：使用超网络（Hypernetwork）生成混合网络的权重，输入是全局状态 $s$，权重通过绝对值操作保证非负（满足单调性）。

### 3.3 MAPPO (Yu et al., 2022)

MAPPO 是多智能体版本的 PPO，属于策略梯度方法：

- **Actor**：每个智能体有独立的策略网络 $\pi_i(a_i | o_i)$，基于局部观测做决策
- **Critic**：共享一个中心化的价值网络 $V(s)$，输入全局状态

策略更新使用 PPO 的裁剪目标：

$$\mathcal{L}_i = \mathbb{E}\left[\min\left(\frac{\pi_i^{\text{new}}}{\pi_i^{\text{old}}} A_i, \text{clip}\left(\frac{\pi_i^{\text{new}}}{\pi_i^{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]$$

其中优势函数 $A_i$ 由中心化 Critic 计算。

MAPPO 的优势在于：实现简单、训练稳定、在多个基准上表现优异。实验表明，精心调参的 MAPPO 可以匹配甚至超越更复杂的 MARL 算法。

### 3.4 算法对比

| 算法 | 类型 | 训练范式 | 适用场景 | 可扩展性 |
|------|------|----------|----------|----------|
| IQL | 值函数 | 独立训练 | 简单任务 | 高 |
| QMIX | 值分解 | CTDE | 合作任务 | 中 |
| MAPPO | 策略梯度 | CTDE | 通用 | 中-高 |
| MADDPG | Actor-Critic | CTDE | 连续动作 | 低 |

## 4. 信用分配问题

在合作 MARL 中，所有智能体共享团队奖励，但每个智能体对团队成功的贡献不同。信用分配（Credit Assignment）问题是：如何将团队奖励合理地归因到每个智能体的贡献上？

- **QMIX 的方案**：通过值分解隐式地进行信用分配
- **COMA 的方案**：使用反事实基线——比较"智能体 $i$ 实际动作"和"智能体 $i$ 的默认动作"对团队奖励的影响差异
- **Shapley Value**：从博弈论角度计算每个智能体的边际贡献，但计算复杂度随智能体数量指数增长

## 参考资料

- Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." ICML, 2018.
- Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS, 2022.
- Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS, 2017.
