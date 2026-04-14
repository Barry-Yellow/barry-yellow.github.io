# 多模态学习导论

> 本文系统介绍多模态学习的基本概念、研究动机和主流方法分类，为后续 CLIP、BLIP 等具体模型的学习提供框架性认识。

## 1. 什么是多模态学习

多模态学习（Multimodal Learning）是指让模型同时处理和理解来自多种模态（modality）的数据。常见的模态包括：
- **视觉**：图像、视频
- **语言**：文本、语音
- **结构化数据**：表格、知识图谱

单模态模型只能处理一种类型的数据，而多模态模型需要解决的核心问题是：如何在不同模态之间建立语义关联，使模型能够跨模态理解和推理。

## 2. 为什么需要多模态

人类的认知本身就是多模态的——我们同时通过视觉、听觉、语言等多个通道感知世界，不同通道的信息相互补充和验证。多模态学习的动机包括：

**信息互补性**：不同模态提供互补的信息。例如，图像提供空间结构信息，文本提供抽象语义信息。单独使用任何一种模态都会丢失另一种模态的信息。

**鲁棒性**：当某个模态的信息缺失或有噪声时，其他模态可以提供补偿。

**泛化能力**：多模态预训练模型（如 CLIP）展现出了强大的零样本迁移能力，能够处理训练时未见过的类别和任务。

**实际应用需求**：图文检索、视觉问答（VQA）、图像描述生成（Image Captioning）、文本到图像生成等任务本质上都是多模态问题。

## 3. 多模态学习的核心挑战

### 3.1 表示学习 (Representation)
如何将不同模态的数据映射到一个共享的语义空间？不同模态的数据结构差异很大（图像是像素矩阵，文本是离散 token 序列），需要设计合适的编码器和对齐策略。

### 3.2 对齐 (Alignment)
如何建立不同模态元素之间的对应关系？例如，图像中的某个区域对应文本中的哪个词或短语。对齐可以是显式的（如注意力机制）或隐式的（如对比学习）。

### 3.3 融合 (Fusion)
如何将不同模态的信息有效地组合起来？主要策略包括：
- **早期融合**：在特征提取阶段就将多模态信息合并
- **晚期融合**：各模态独立提取特征后再合并决策
- **中间融合**：在模型的中间层通过 cross-attention 等机制交互

### 3.4 生成 (Generation)
如何从一种模态生成另一种模态的内容？如文本生成图像（DALL-E、Stable Diffusion）、图像生成描述（Image Captioning）。

## 4. 主流方法分类

### 4.1 对比学习方法
**代表模型**：CLIP、ALIGN、SigLIP

核心思想：通过对比损失将匹配的图文对拉近、不匹配的推远，从而在共享空间中对齐不同模态的表示。

CLIP 的对比损失（InfoNCE）：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_i, t_j)/\tau)}$$

对比学习方法的优势在于：不需要细粒度的标注（只需图文对），可以利用大规模互联网数据进行预训练，且天然支持零样本迁移。

### 4.2 生成式方法
**代表模型**：BLIP、BLIP-2、Flamingo、LLaVA

核心思想：将多模态理解建模为条件生成问题。给定图像，生成对应的文本描述；或给定文本指令和图像，生成回答。

BLIP 系列的关键创新在于统一了理解和生成任务：
- BLIP 提出 CapFilt 机制，利用噪声网络数据进行高效预训练
- BLIP-2 通过 Q-Former 桥接冻结的视觉编码器和语言模型，大幅降低训练成本

### 4.3 扩散模型方法
**代表模型**：DALL-E 2、Stable Diffusion、Imagen

核心思想：通过前向加噪和反向去噪过程实现图像生成。文本条件通过 cross-attention 注入去噪网络，控制生成内容。

### 4.4 自回归方法
**代表模型**：GPT-4V、Gemini、Chameleon

核心思想：将图像 token 化后与文本 token 统一处理，用自回归方式同时建模理解和生成。

## 5. 评估指标

| 指标 | 用途 | 计算方式 |
|------|------|----------|
| **FID** (Frechet Inception Distance) | 图像生成质量 | 比较生成图像和真实图像在 Inception 网络特征空间中的分布距离，越小越好 |
| **CLIP Score** | 图文匹配度 | 用 CLIP 模型计算生成图像与文本描述的余弦相似度，越高越好 |
| **CIDEr** | 图像描述质量 | 基于 TF-IDF 加权的 n-gram 匹配 |
| **VQA Accuracy** | 视觉问答准确率 | 生成答案与标准答案的匹配度 |

## 6. 学习路线建议

建议按以下顺序阅读本部分的笔记：
1. **CLIP** → 理解对比学习和多模态对齐的基本范式
2. **BLIP** → 理解如何统一多模态理解和生成
3. **BLIP-2** → 理解如何高效地连接视觉和语言模型
4. **Diffusion** → 理解扩散模型的数学原理
5. **AutoAggressive** → 对比自回归和扩散两种生成范式

## 参考资料

- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021.
- Li et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML, 2022.
- Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML, 2023.
- Baltrušaitis et al. "Multimodal Machine Learning: A Survey and Taxonomy." TPAMI, 2019.
