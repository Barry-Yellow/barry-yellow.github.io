# Informer: 高效长序列时间序列预测

> Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI, 2021.

## 1. 背景与动机

标准 Transformer 应用于长序列时间序列预测时面临三个核心问题：
- **二次计算复杂度**：自注意力的 $O(L^2)$ 复杂度使得处理长序列（如数千个时间步）代价过高
- **高内存占用**：编码器-解码器架构需要存储大量中间状态
- **逐步解码慢**：解码器自回归生成预测序列，速度随预测长度线性增长

Informer 的目标是在保持 Transformer 建模能力的同时，将复杂度降低到 $O(L \log L)$。

## 2. 核心方法

### 2.1 ProbSparse Self-Attention

Informer 的核心创新是 ProbSparse 注意力机制。其观察是：在标准自注意力中，大部分 query 的注意力分布接近均匀分布（即对所有 key 的关注度差不多），只有少数 query 有"尖锐"的注意力分布。

**稀疏性度量**：定义 query $q_i$ 的稀疏性为其注意力分布与均匀分布的 KL 散度：

$$M(q_i, K) = \ln \sum_{j=1}^{L_K} e^{q_i k_j^T / \sqrt{d}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^T}{\sqrt{d}}$$

$M$ 值越大，说明该 query 的注意力分布越"尖锐"，包含的信息量越大。

**Top-u 选择**：只保留 $M$ 值最大的 $u = c \cdot \ln L_Q$ 个 query 进行完整的注意力计算，其余 query 用均值替代。这将复杂度从 $O(L^2)$ 降低到 $O(L \ln L)$。

### 2.2 Self-Attention Distilling

为了进一步减少特征图的尺寸，Informer 在编码器的相邻注意力层之间加入蒸馏操作：

$$X_{j+1} = \text{MaxPool}(\text{ELU}(\text{Conv1d}(X_j^{\text{Attention}})))$$

每一层将序列长度减半，形成金字塔结构。这不仅减少了计算量，还起到了多尺度特征提取的作用。

### 2.3 Generative Style Decoder

传统 Transformer 解码器逐步生成预测序列（自回归），Informer 改为一次性生成整个预测序列：

- 将目标序列的前半部分（已知的）作为"起始 token"
- 后半部分用占位符填充
- 解码器通过一次前向传播同时预测所有未来时间步

这种方式将解码的时间复杂度从 $O(L)$ 降低到 $O(1)$。

## 3. 实验结果

在 ETTh、ETTm、ECL、Weather 等长序列预测基准上，Informer 在预测长度为 720 步时仍保持较好的性能，而标准 Transformer 在 200 步后性能急剧下降。

## 参考资料

- Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI, 2021.

---

# PatchTST: 基于 Patch 的时间序列 Transformer

> Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR, 2023.

## 1. 背景与动机

将 Transformer 应用于时间序列预测时，一个关键问题是：如何将连续的时间序列转化为 Transformer 能处理的 token 序列？

早期方法（如 Informer）将每个时间步作为一个 token，这导致：
- 序列长度等于时间步数，长序列的计算量大
- 单个时间步包含的信息量有限，语义粒度太细

PatchTST 借鉴了 ViT（Vision Transformer）中 patch 的思想：将时间序列分割为固定长度的 patch，每个 patch 作为一个 token。

## 2. 核心方法

### 2.1 Patching

将长度为 $L$ 的时间序列分割为 $N = \lfloor (L - P) / S \rfloor + 1$ 个 patch，其中 $P$ 是 patch 长度，$S$ 是步长（stride）。

例如，对于长度 512 的序列，使用 $P=16, S=8$，得到 63 个 patch。相比逐点 tokenization 的 512 个 token，序列长度大幅缩短。

每个 patch 通过线性投影映射到 $d$-维嵌入空间：

$$x_i^{(p)} = W_p \cdot \text{patch}_i + b_p$$

### 2.2 Channel Independence

PatchTST 的另一个关键设计是通道独立（Channel Independence）：对于多变量时间序列，每个变量独立地通过同一个 Transformer 处理，而非将所有变量拼接在一起。

**直觉**：不同变量的时间模式可能差异很大（如温度和湿度），强制它们共享注意力可能引入噪声。通道独立让模型专注于每个变量自身的时间模式。

**实验验证**：在大多数基准上，通道独立的效果优于通道混合（Channel Mixing），尤其是在变量数较多时。

### 2.3 整体架构

```
输入序列 (B, C, L)
    ↓ Patching
Patch 序列 (B*C, N, P)  [通道独立]
    ↓ Linear Projection
Token 嵌入 (B*C, N, d)
    ↓ + Positional Encoding
    ↓ Transformer Encoder (多层)
编码输出 (B*C, N, d)
    ↓ Flatten + Linear
预测输出 (B, C, T)
```

### 2.4 自监督预训练

PatchTST 还支持自监督预训练：随机遮盖部分 patch，让模型预测被遮盖的 patch（类似 BERT 的 MLM）。预训练后在下游预测任务上微调，可以进一步提升性能。

## 3. 实验结果

PatchTST 在 8 个主流时间序列预测基准上取得了当时的 SOTA，相比 Informer、FEDformer 等方法有显著提升。关键发现：

- Patch 大小 $P=16$ 在大多数数据集上效果最好
- 通道独立在 7/8 个数据集上优于通道混合
- 自监督预训练在小数据集上提升明显（5-10%）

## 4. PatchTST 的意义

PatchTST 证明了一个重要观点：时间序列预测中，简单的架构设计（patching + channel independence）比复杂的注意力改进（如 ProbSparse、频域注意力）更有效。这启发了后续一系列"简单即有效"的工作。

## 参考资料

- Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR, 2023.
- Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI, 2021.
