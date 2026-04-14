# 语言模型 (Language Model)

本部分整理了自然语言处理和大语言模型相关的基础知识笔记，从经典 NLP 方法到现代大模型技术。

## 笔记内容

- **NLP 基础知识**：词嵌入（Word2Vec、GloVe）、上下文相关表示（ELMo）等经典 NLP 技术
- **Transformer**：多头自注意力、位置编码、前馈网络，编码器-解码器结构
- **BERT**：掩码语言模型（MLM）+ 下一句预测（NSP），双向语言表示的预训练范式
- **大模型基础**：Prompt Engineering、RAG（检索增强生成）、微调（LoRA、SFT、RLHF）
- **强化学习基础**：REINFORCE、Actor-Critic、PPO，理解 RLHF 的前置知识
- **MoE (Mixture of Experts)**：门控机制、稀疏激活、负载均衡，GShard/Switch Transformer/Mixtral

## 领域经典论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| Word2Vec (Mikolov et al.) | 2013 | 开创性的词嵌入方法，Skip-gram 和 CBOW |
| Attention Is All You Need (Vaswani et al.) | 2017 | 提出 Transformer 架构，奠定现代 NLP 基础 |
| BERT (Devlin et al.) | 2019 | 双向预训练语言模型，MLM + NSP |
| GPT-2/3 (Radford et al. / Brown et al.) | 2019/2020 | 自回归语言模型，展示 few-shot 和 in-context learning 能力 |
| LoRA (Hu et al.) | 2022 | 低秩适配的参数高效微调方法 |
| InstructGPT (Ouyang et al.) | 2022 | RLHF 对齐方法，从人类反馈中学习 |
| Switch Transformer (Fedus et al.) | 2022 | 简化 MoE 门控为 Top-1 路由，万亿参数模型 |
| Mixtral (Mistral AI) | 2024 | 开源 MoE 模型，8x7B 专家，性能匹配 GPT-3.5 |

[](_sidebar.md ':include')
