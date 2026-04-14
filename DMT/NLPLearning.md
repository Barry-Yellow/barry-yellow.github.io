# NLP 基础学习笔记

> 本文整理了 NLP 领域的基础知识，重点关注从词嵌入到注意力机制的技术演进，为理解多模态模型中的文本处理模块提供基础。

## 1. 词嵌入 (Word Embedding)

词嵌入是 NLP 的基石，其核心思想是将离散的词汇映射到连续的向量空间中，使得语义相近的词在向量空间中距离也相近。

### 1.1 静态词嵌入

**Word2Vec** (Mikolov et al., 2013) 提出了两种训练范式：
- **CBOW (Continuous Bag of Words)**：用上下文词预测中心词。给定窗口内的上下文词 $w_{t-k}, ..., w_{t+k}$，最大化中心词 $w_t$ 的条件概率。
- **Skip-gram**：用中心词预测上下文词。给定中心词 $w_t$，最大化上下文词的条件概率。

训练目标本质上是一个 softmax 分类问题，但由于词表 $V$ 通常很大（数万到数十万），直接计算 softmax 代价过高，因此引入了负采样（Negative Sampling）或层次 softmax 来近似。

**GloVe** (Pennington et al., 2014) 从全局共现矩阵出发，通过矩阵分解学习词向量。其损失函数为：

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

其中 $X_{ij}$ 是词 $i$ 和词 $j$ 的共现次数，$f$ 是权重函数，用于抑制高频词对的过大影响。

静态词嵌入的局限性在于：每个词只有一个固定的向量表示，无法处理一词多义问题（如"bank"既可以是银行也可以是河岸）。

### 1.2 上下文相关词嵌入

**ELMo** (Peters et al., 2018) 首次提出了上下文相关的词表示。其核心思想是：先用双向 LSTM 语言模型进行预训练，然后将各层的隐藏状态加权求和作为词的上下文表示。

ELMo 的预训练目标是双向语言模型的联合似然：

$$\sum_{k=1}^{N} (\log p(t_k | t_1, ..., t_{k-1}; \Theta) + \log p(t_k | t_{k+1}, ..., t_N; \Theta))$$

前向 LSTM 从左到右编码，后向 LSTM 从右到左编码，最终将两个方向的隐藏状态拼接。ELMo 的关键创新在于：不只使用最后一层的输出，而是将所有层的表示加权组合：

$$\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j h_{k,j}$$

其中 $s_j$ 是可学习的层权重，$\gamma$ 是缩放因子。不同层捕获不同层次的语言信息：底层偏向句法特征，高层偏向语义特征。

## 2. RNN/LSTM 在 NLP 中的应用

### 2.1 循环神经网络 (RNN)

RNN 通过隐藏状态 $h_t$ 在时间步之间传递信息：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

RNN 的核心问题是梯度消失/爆炸：当序列较长时，反向传播的梯度会随时间步指数衰减或增长，导致模型难以学习长距离依赖。

### 2.2 LSTM (Long Short-Term Memory)

LSTM 通过门控机制解决了 RNN 的梯度消失问题。其核心是引入了记忆单元 $C_t$ 和三个门：

- **遗忘门** $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$：决定丢弃哪些旧信息
- **输入门** $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$：决定存储哪些新信息
- **输出门** $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$：决定输出哪些信息

记忆单元更新：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

LSTM 在 NLP 中的典型应用包括：语言模型、机器翻译（Seq2Seq）、文本分类、命名实体识别等。在多模态场景中，LSTM 常用于处理文本序列或视频帧序列。

### 2.3 GRU (Gated Recurrent Unit)

GRU 是 LSTM 的简化版本，将遗忘门和输入门合并为更新门 $z_t$，同时合并了记忆单元和隐藏状态。参数更少，训练更快，在许多任务上效果与 LSTM 相当。

## 3. 注意力机制在 NLP 中的演进

### 3.1 Seq2Seq + Attention

Bahdanau et al. (2015) 首次将注意力机制引入 NLP。在 Seq2Seq 模型中，解码器在生成每个词时，不再只依赖编码器的最后一个隐藏状态，而是对编码器所有时间步的隐藏状态进行加权求和：

$$c_t = \sum_{j} \alpha_{tj} h_j, \quad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_k \exp(e_{tk})}$$

其中 $e_{tj} = a(s_{t-1}, h_j)$ 是对齐函数，衡量解码器状态 $s_{t-1}$ 与编码器状态 $h_j$ 的相关性。

### 3.2 Self-Attention 与 Transformer

Vaswani et al. (2017) 提出的 Transformer 完全抛弃了 RNN 结构，仅使用自注意力机制。自注意力的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Transformer 的出现彻底改变了 NLP 的技术路线，后续的 BERT、GPT 系列都建立在 Transformer 架构之上。详细的 Transformer 架构分析见 [Transformer 笔记](LM/Transformer.md)。

### 3.3 从 NLP 到多模态

注意力机制在多模态学习中扮演了关键角色：
- **Cross-Attention**：在 BLIP、BLIP-2 等模型中，用于实现图像特征和文本特征之间的交互。
- **CLIP 的对比学习**：虽然不直接使用 cross-attention，但通过对比损失实现了图文语义空间的对齐。

## 参考资料

- Mikolov et al. "Efficient Estimation of Word Representations in Vector Space." ICLR, 2013.
- Peters et al. "Deep contextualized word representations." NAACL, 2018.
- Vaswani et al. "Attention Is All You Need." NeurIPS, 2017.
