## BERT (Bidirectional Encoder Representations from Transformers)

> [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)

BERT 是 Google 于 2018 年提出的预训练语言模型，通过在大规模无标注文本上进行双向预训练，然后在下游任务上微调，刷新了 11 项 NLP 任务的 SOTA。BERT 的核心贡献在于证明了双向上下文建模的重要性。

### 1. 核心思想

在 BERT 之前，语言模型的预训练主要有两种方式：
- **单向语言模型**（如 GPT）：从左到右预测下一个词，只能利用左侧上下文
- **浅层双向**（如 ELMo）：分别训练前向和后向 LSTM，然后拼接，但两个方向的信息没有深度交互

BERT 的关键创新是：使用 Transformer 编码器（而非解码器），通过掩码语言模型（MLM）任务实现真正的深度双向预训练。每一层的每个 token 都能同时关注左右两侧的所有 token。

### 2. 模型架构

BERT 直接使用 Transformer 的编码器部分：
- **BERT-Base**：12 层，768 维隐藏层，12 个注意力头，110M 参数
- **BERT-Large**：24 层，1024 维隐藏层，16 个注意力头，340M 参数

输入表示由三部分相加：
- **Token Embedding**：WordPiece 分词后的 token 嵌入
- **Segment Embedding**：区分句子 A 和句子 B（用于句对任务）
- **Position Embedding**：可学习的位置嵌入（最大长度 512）

特殊 token：`[CLS]` 放在序列开头，其最终隐藏状态用于分类任务；`[SEP]` 用于分隔两个句子。

### 3. 预训练任务

#### 3.1 掩码语言模型 (Masked Language Model, MLM)

随机遮盖输入中 15% 的 token，让模型预测被遮盖的词。对于被选中的 token：
- 80% 的概率替换为 `[MASK]`
- 10% 的概率替换为随机词
- 10% 的概率保持不变

这种设计是为了缓解预训练和微调之间的不一致（微调时没有 `[MASK]` token）。

MLM 的损失函数只计算被遮盖位置的交叉熵：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log p(x_i | x_{\backslash i})$$

#### 3.2 下一句预测 (Next Sentence Prediction, NSP)

给定句子对 (A, B)，预测 B 是否是 A 的下一句。50% 的训练样本中 B 是真正的下一句（标签 IsNext），50% 是随机采样的句子（标签 NotNext）。

NSP 任务使用 `[CLS]` token 的输出进行二分类。后续研究（如 RoBERTa）发现 NSP 任务的贡献有限，甚至可能有害。

### 4. 微调策略

BERT 的微调非常简单：在预训练模型的基础上，根据下游任务添加一个任务特定的输出层，然后端到端微调所有参数。

| 任务类型 | 输入格式 | 输出方式 |
|----------|----------|----------|
| 单句分类（情感分析） | `[CLS] 句子 [SEP]` | `[CLS]` 输出 → 分类层 |
| 句对分类（自然语言推理） | `[CLS] 句子A [SEP] 句子B [SEP]` | `[CLS]` 输出 → 分类层 |
| 序列标注（NER） | `[CLS] 句子 [SEP]` | 每个 token 输出 → 标签 |
| 阅读理解（SQuAD） | `[CLS] 问题 [SEP] 段落 [SEP]` | 预测答案的起止位置 |

### 5. BERT 的影响

BERT 开创了"预训练 + 微调"的 NLP 范式，后续的 RoBERTa、ALBERT、DeBERTa 等模型都是在 BERT 基础上的改进。GPT 系列则走了另一条路——用更大的单向模型 + 提示学习（Prompt）来替代微调。

## 参考资料

- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL, 2019.
- Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv, 2019.
