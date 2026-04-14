# MoE (Mixture of Experts)

> 混合专家模型是大模型扩展的重要技术路线，通过稀疏激活实现"大模型容量、小模型计算量"的目标。

## 1. 核心思想

传统的 Dense 模型（如标准 Transformer）在推理时激活所有参数，计算量与参数量成正比。MoE 的核心思想是：将模型的某些层（通常是 FFN 层）替换为多个"专家"网络，每次推理时只激活其中少数几个专家，从而在大幅增加模型容量的同时保持计算量基本不变。

## 2. 基本架构

MoE 层由两部分组成：

### 2.1 专家网络 (Experts)
$N$ 个结构相同但参数不同的前馈网络 $E_1, E_2, ..., E_N$。每个专家通常是一个标准的 FFN：

$$E_i(x) = W_2^{(i)} \cdot \text{ReLU}(W_1^{(i)} x + b_1^{(i)}) + b_2^{(i)}$$

### 2.2 门控网络 (Gating Network)
门控网络决定每个 token 应该被路由到哪些专家。最简单的 Top-K 门控：

$$G(x) = \text{TopK}(\text{softmax}(W_g x + \epsilon))$$

其中 $\epsilon$ 是可选的噪声项（用于训练时的负载均衡），TopK 操作只保留概率最高的 $K$ 个专家的权重，其余置零。

MoE 层的输出是被选中专家输出的加权和：

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

实际计算中，只需要计算被选中的 $K$ 个专家（通常 $K=1$ 或 $K=2$），未被选中的专家不参与计算。

## 3. 关键技术问题

### 3.1 负载均衡 (Load Balancing)

MoE 训练中最大的挑战是负载不均衡：门控网络倾向于将大部分 token 路由到少数几个"热门"专家，导致其他专家得不到充分训练（"专家坍塌"问题）。

**辅助损失**（Auxiliary Loss）是最常用的解决方案。Switch Transformer 使用的负载均衡损失：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是门控网络分配给专家 $i$ 的平均概率，$\alpha$ 是超参数。这个损失鼓励 $f_i$ 和 $P_i$ 都趋向均匀分布 $1/N$。

### 3.2 专家容量 (Expert Capacity)

为了保证计算效率，每个专家在每个 batch 中能处理的 token 数量有上限（容量因子 $C$）。超出容量的 token 会被丢弃或路由到其他专家。

### 3.3 通信开销

在分布式训练中，不同专家通常分布在不同的设备上，token 路由需要跨设备通信（All-to-All），这是 MoE 训练的主要瓶颈之一。

## 4. 代表性模型

### 4.1 GShard (Lepikhin et al., 2021)
- 首次将 MoE 扩展到 6000 亿参数的 Transformer
- 提出 Top-2 门控和随机路由策略
- 每个 MoE 层有 2048 个专家

### 4.2 Switch Transformer (Fedus et al., 2022)
- 简化为 Top-1 路由（每个 token 只选一个专家），降低通信开销
- 提出容量因子和改进的负载均衡损失
- 在相同计算预算下，训练速度比 T5 快 7 倍

### 4.3 Mixtral 8x7B (Mistral AI, 2024)
- 8 个专家，每次激活 2 个，总参数 47B，活跃参数约 13B
- 在多数基准上超越 LLaMA 2 70B，但推理成本只有其 1/5
- 证明了 MoE 在开源大模型中的实用性

### 4.4 DeepSeek-MoE (DeepSeek, 2024)
- 提出细粒度专家分割和共享专家机制
- 将部分专家设为"共享专家"（所有 token 都经过），其余为"路由专家"
- 在更少的计算量下达到了 Dense 模型的性能

## 5. MoE 的优势与局限

| 方面 | 优势 | 局限 |
|------|------|------|
| 计算效率 | 参数量大但计算量小，推理速度快 | 内存占用大（所有专家参数都需加载） |
| 扩展性 | 容易通过增加专家数量扩展模型容量 | 分布式训练的通信开销大 |
| 性能 | 相同计算预算下通常优于 Dense 模型 | 负载均衡和专家坍塌问题需要精心设计 |
| 微调 | — | 微调时可能只更新少数专家，泛化性受限 |

## 参考资料

- Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR, 2017.
- Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR, 2022.
- Jiang et al. "Mixtral of Experts." arXiv, 2024.
