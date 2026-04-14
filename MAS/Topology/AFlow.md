# AFlow: Automating Agentic Workflow Generation

> [!NOTE|label:重点]
> 
> AFlow 将多智能体工作流建模为代码节点组成的图，通过蒙特卡洛树搜索（MCTS）自动搜索最优工作流结构，无需人工设计。
>
> 核心思想：用 LLM 作为优化器，在代码空间中搜索和优化工作流。

Zhang et al., ICLR 2025
https://arxiv.org/abs/2410.10762
https://github.com/geekan/MetaGPT

## 一、研究背景与问题

现有的多智能体工作流设计面临两个核心问题：
1. **人工设计成本高**：Chain-of-Thought、Self-Refine、Multi-Agent Debate 等工作流都需要人类专家手动设计，难以适配不同任务。
2. **自动化方法效率低**：已有的自动化方法（如 DSPy、GPTSwarm）要么搜索空间受限于预定义的操作符，要么需要大量 LLM 调用进行迭代优化。

AFlow 的目标是：在一个开放的搜索空间中，自动发现高效的多智能体工作流。

## 二、核心方法

### 1. 工作流表示：代码即工作流

AFlow 将工作流表示为 Python 代码，其中每个节点（Node）是一个 LLM 操作单元。预定义了几种基本节点类型：
- **LLM Node**：调用 LLM 生成回答
- **Ensemble Node**：多个 LLM 输出的投票/聚合
- **Review Node**：对输出进行评审和反馈
- **Revise Node**：根据反馈修改输出

工作流的搜索空间是这些节点的任意组合和连接方式，用 Python 代码表达。这种表示的优势在于：搜索空间是开放的（不限于预定义的拓扑模板），且生成的工作流可以直接执行。

### 2. 搜索算法：MCTS + LLM 优化器

AFlow 使用蒙特卡洛树搜索（MCTS）来探索工作流空间：

**选择（Selection）**：从根节点出发，根据 UCB（Upper Confidence Bound）公式选择子节点：

$$\text{UCB}(v) = \bar{X}_v + c \sqrt{\frac{\ln N_{\text{parent}}}{N_v}}$$

其中 $\bar{X}_v$ 是节点的平均得分，$N_v$ 是访问次数，$c$ 是探索系数。

**扩展（Expansion）**：使用 LLM 作为优化器，基于当前工作流代码和历史反馈，生成新的工作流变体。LLM 可以修改节点的 prompt、调整节点连接方式、增删节点等。

**评估（Evaluation）**：在验证集上运行生成的工作流，计算任务准确率作为得分。

**回传（Backpropagation）**：将评估得分回传到路径上的所有节点，更新统计信息。

### 3. 经验库 (Experience Pool)

AFlow 维护一个经验库，记录历史搜索中发现的高分工作流和低分工作流。在扩展阶段，LLM 优化器可以参考经验库中的成功案例和失败案例，避免重复探索无效方向。

## 三、实验结果

在 6 个基准任务（数学推理、代码生成、常识问答等）上的实验表明：
- AFlow 发现的工作流在多数任务上超越了人工设计的工作流（如 CoT、Self-Refine、Multi-Agent Debate）
- 搜索成本显著低于其他自动化方法（如 GPTSwarm 需要的 LLM 调用次数是 AFlow 的 5-10 倍）
- 发现的工作流具有可解释性——可以直接阅读生成的 Python 代码理解工作流逻辑

## 四、与其他方法的对比

| 方法 | 搜索空间 | 优化方式 | 可解释性 |
|------|----------|----------|----------|
| DSPy | 预定义操作符组合 | 贝叶斯优化 | 中 |
| GPTSwarm | 图结构 | 梯度优化 | 低 |
| ADAS | 代码空间 | LLM 生成 | 高 |
| **AFlow** | 代码空间 | MCTS + LLM | 高 |

AFlow 的核心优势在于：MCTS 提供了系统化的搜索策略（平衡探索与利用），而代码表示提供了开放的搜索空间和良好的可解释性。

> @inproceedings{zhang2025aflow,
  title={AFlow: Automating Agentic Workflow Generation},
  author={Zhang, Jiayi and Xiang, Jinyu and Yu, Zhaoyang and others},
  booktitle={ICLR},
  year={2025}
}
