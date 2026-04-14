# 频域 Transformer (Frequency Transformer)

本部分记录了将频域分析与 Transformer 架构结合的一系列模型笔记，涵盖计算机视觉和时间序列预测两个方向。

Transformer 的自注意力机制在时域上建模全局依赖关系，但对信号的频率结构缺乏显式建模能力。频域 Transformer 的核心思路是：在 Transformer 框架中引入傅里叶变换、小波变换或 DCT 等频域工具，使模型能同时捕获时域和频域特征。笔记包含以下内容：

- **频率 + Transformer 综述**：梳理 FNet、GFNet、SpectFormer 等将频域操作引入 Transformer 的代表性工作，主要面向计算机视觉方向。
- **SpectFormer**：提出层次化的谱注意力与自注意力结合方案，在 ViT 框架中用频率域的全局滤波替代部分自注意力层。
- **AutoFormer**：基于序列周期性设计自相关机制，用频域中的周期性信息替代传统的点积注意力，专门面向时间序列预测。
- **FEDformer**：在频域中进行注意力计算，利用傅里叶变换和小波变换实现频率增强的 Transformer。
- **FECAM (DCT Transformer)**：用离散余弦变换（DCT）改进 SE（Squeeze-and-Excitation）架构中的通道注意力机制。
- **特征的不同视角与数据预处理**：从 Barra 模型等角度讨论特征构建方法。（详见资产定价部分）

[](_sidebar.md ':include')
