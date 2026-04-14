# 多智能体强化学习 (Multi-Agent Reinforcement Learning)

本部分记录了多智能体强化学习（MARL）方向的学习笔记。MARL 研究多个智能体在共享环境中如何通过交互学习协作或竞争策略。

与单智能体强化学习相比，MARL 面临的核心挑战包括：环境非平稳性（其他智能体的策略在不断变化）、信用分配问题（如何将团队奖励归因到个体贡献）、通信与协调（智能体之间如何高效交换信息）。

## 笔记内容

- **MARL 基础概念**：从单智能体 RL 到多智能体的扩展，包括完全合作、完全竞争、混合博弈等问题设定，以及集中式训练分散式执行（CTDE）范式。
- **经典算法笔记**：IQL、QMIX、MAPPO、MADDPG 等代表性算法的原理和实现。
- **安全 MARL 综述**：在多智能体场景下引入安全约束的研究综述。

## 领域经典论文与基准

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| IQL (Tan) | 1993 | 最早的独立 Q-Learning 多智能体方法 |
| COMA (Foerster et al.) | 2018 | 反事实基线解决信用分配问题 |
| QMIX (Rashid et al.) | 2018 | 值分解 + 单调性约束，合作 MARL 的里程碑 |
| MADDPG (Lowe et al.) | 2017 | 多智能体 Actor-Critic，支持混合合作竞争场景 |
| MAPPO (Yu et al.) | 2022 | 证明精心调参的 PPO 在合作 MARL 中极具竞争力 |
| QPLEX (Wang et al.) | 2021 | 基于对偶约束的值分解，突破 QMIX 的单调性限制 |
| MAT (Wen et al.) | 2022 | 将 Transformer 引入 MARL，用序列建模处理多智能体决策 |

经典基准环境：[StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac)、[Google Research Football](https://github.com/google-research/football)、[MPE](https://pettingzoo.farama.org/environments/mpe/)

[](_sidebar.md ':include')
