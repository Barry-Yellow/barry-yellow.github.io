# MIPRO: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs

> [!NOTE|label:重点]
> 
> MIPRO 是 DSPy 框架下的提示词优化方法，能够自动为多阶段 LLM 程序中的每个模块优化指令（instruction）和示例（demonstration）。
>
> 核心思想：将提示词优化建模为贝叶斯优化问题，联合优化多个模块的提示词。

Opsahl-Ong et al., EMNLP 2024
https://arxiv.org/abs/2406.11695

## 一、研究背景与问题

现代 LLM 应用通常由多个模块组成（如 RAG 系统中的检索模块 + 生成模块），每个模块都需要精心设计的提示词。现有的提示词优化方法存在以下问题：

1. **独立优化**：大多数方法（如 APE、OPRO）只优化单个模块的提示词，忽略了模块之间的依赖关系
2. **缺乏示例优化**：很多方法只优化指令文本，不考虑 few-shot 示例的选择
3. **搜索效率低**：暴力搜索所有可能的提示词组合在多模块场景下计算量爆炸

MIPRO 的目标是：在多阶段 LLM 程序中，联合优化所有模块的指令和示例。

## 二、核心方法

### 1. 问题形式化

一个多阶段 LLM 程序 $P$ 由 $M$ 个模块组成：$P = (m_1, m_2, ..., m_M)$。每个模块 $m_i$ 的提示词由两部分组成：
- **指令** $\text{inst}_i$：描述模块任务的自然语言文本
- **示例** $\text{demo}_i$：输入-输出示例集合

优化目标是找到使整个程序在验证集上性能最优的提示词组合：

$$(\text{inst}^*, \text{demo}^*) = \arg\max \sum_{(x,y) \in D_{\text{val}}} \text{score}(P(x; \text{inst}, \text{demo}), y)$$

### 2. 指令生成 (Instruction Proposal)

MIPRO 使用 LLM 作为"元提示词生成器"，基于以下信息生成候选指令：
- 程序的整体描述和模块功能
- 训练数据中的输入-输出示例
- 历史优化中表现好的指令

对每个模块生成 $K$ 个候选指令，形成指令候选池。

### 3. 示例选择 (Demonstration Selection)

对于每个模块，MIPRO 从训练集中选择最有代表性的示例。选择策略考虑：
- 示例的多样性（覆盖不同类型的输入）
- 示例与当前查询的相关性
- 示例在 bootstrap 过程中的成功率（只选择模型能正确处理的示例）

### 4. 贝叶斯优化 (Bayesian Optimization)

MIPRO 使用 TPE（Tree-structured Parzen Estimator）进行贝叶斯优化，在指令候选池和示例候选池中搜索最优组合：

1. **初始化**：随机采样若干提示词组合，在验证集上评估
2. **建模**：基于历史评估结果，建立"提示词组合 → 性能"的概率模型
3. **采样**：从概率模型中采样最有可能提升性能的新组合
4. **评估**：在验证集上评估新组合，更新概率模型
5. 重复步骤 3-4 直到预算用完

贝叶斯优化的优势在于：能够在有限的评估预算内高效搜索高维组合空间，且自然地处理模块间的依赖关系。

## 三、实验结果

在多个多阶段 LLM 程序上的实验（包括 RAG、多跳问答、数学推理等）表明：
- MIPRO 优化后的提示词比人工设计的提示词平均提升 10-20% 的准确率
- 联合优化多个模块的效果显著优于独立优化每个模块
- 贝叶斯优化比随机搜索和网格搜索更高效

## 四、与其他提示词优化方法的对比

| 方法 | 优化对象 | 多模块支持 | 示例优化 | 搜索策略 |
|------|----------|------------|----------|----------|
| APE | 单模块指令 | 否 | 否 | LLM 生成 + 评估 |
| OPRO | 单模块指令 | 否 | 否 | LLM 迭代优化 |
| DSPy BootstrapFewShot | 示例 | 是 | 是 | Bootstrap |
| **MIPRO** | 指令 + 示例 | 是 | 是 | 贝叶斯优化 |

MIPRO 的核心贡献在于：首次在多阶段 LLM 程序中实现了指令和示例的联合优化，并通过贝叶斯优化高效搜索组合空间。

> @inproceedings{opsahl-ong-etal-2024-optimizing,
  title={Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs},
  author={Opsahl-Ong, Krista and Ryan, Michael J and others},
  booktitle={EMNLP},
  year={2024}
}
