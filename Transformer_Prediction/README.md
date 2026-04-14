# Transformer 预测 (Transformer Prediction)

本部分记录了 Transformer 在时间序列预测中的应用笔记。Transformer 凭借自注意力机制捕获长距离依赖的能力，已成为时间序列预测领域的重要架构。

与传统 RNN/LSTM 方法相比，Transformer 在时间序列预测中的优势在于：能够并行处理整个序列、通过注意力机制直接建模任意两个时间步之间的关系、避免了梯度消失问题。但也面临二次计算复杂度、对序列顺序信息的建模不够自然等挑战。

笔记包含以下内容：

- **时间序列预测经典模型**：梳理了 Transformer 在时间序列预测中的代表性工作，包括 Informer、PatchTST 等模型的核心思想和技术贡献。
- **特征的不同视角与数据预处理**：讨论时间序列特征的构建方式和预处理方法。

[](_sidebar.md ':include')
