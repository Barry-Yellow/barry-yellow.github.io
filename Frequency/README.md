# 频域分析与 Transformer 预测

本部分整合了频域分析基础、频域 Transformer 模型和 Transformer 时间序列预测三个方向的笔记。

## 频域基础

频域分析的核心思想是将时域信号通过傅里叶变换、小波变换等工具转换到频域，揭示信号中隐藏的周期性成分和多尺度结构。

- **自相关系数与自谱密度**：从自相关函数出发推导功率谱，扩展到多变量情形下的互协方差和共谱密度
- **时序数据平稳方法**：对比 EMD、X11 季节调整、差分等平稳化方法的适用场景

## 频域 Transformer

在 Transformer 框架中引入傅里叶变换、小波变换或 DCT 等频域工具，使模型能同时捕获时域和频域特征。

- **FNet / GFNet / SpectFormer**：将频域操作引入 Transformer 的代表性工作（CV 方向）
- **AutoFormer**：基于序列周期性设计自相关机制，用频域周期性信息替代点积注意力
- **FEDformer**：在频域中进行注意力计算，利用傅里叶变换和小波变换实现频率增强
- **FECAM**：用 DCT 改进 SE 架构中的通道注意力机制

## Transformer 时序预测

Transformer 凭借自注意力机制捕获长距离依赖的能力，已成为时间序列预测领域的重要架构。

- **Informer**：ProbSparse 注意力 + 自注意力蒸馏 + 生成式解码器，解决长序列预测的效率问题
- **PatchTST**：将时间序列分割为 patch 输入 Transformer，结合通道独立策略，在多个基准上达到 SOTA

[](_sidebar.md ':include')
