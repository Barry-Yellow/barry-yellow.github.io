# LLM 面试知识库目录

本仓库收录大模型相关面试知识点，共9个主题文档。

---

## 📄 文档列表

| 文档 | 主题 | 核心板块 |
|------|------|----------|
| [01_LLM_Basics.md](./01_LLM_Basics.md) | 大模型基础 | Transformer架构、位置编码、注意力变体、MoE、分布式训练、LoRA、FlashAttention、KV Cache、词元化、Self/Cross-Attention、Encoding全览、词表生成、Token向量化、多模态Transformer设计、GLM架构、Prompt工程 |
| [02_VLM.md](./02_VLM.md) | 多模态大模型 | 模态对齐与融合、CLIP、LLaVA、视觉指令微调、视频理解、视觉定位、VLM幻觉问题、VLA模型、Embedding相似度、多模态GRPO |
| [03_RLHF.md](./03_RLHF.md) | 强化学习与对齐 | RLHF流程、奖励模型、PPO/DPO/GRPO全面对比、偏好学习、KL惩罚、信用分配、熵崩问题、CoT必要性与质量评估、持续学习、Human-in-the-loop标注、RL落地 |
| [04_Agent.md](./04_Agent.md) | Agent智能体 | ReAct框架、规划能力(CoT/ToT/GoT)、记忆设计、工具调用、多智能体系统、Agent安全、MCP协议、任务拆分、长上下文、模型能力vs框架设计 |
| [05_RAG.md](./05_RAG.md) | 检索增强生成 | RAG原理、检索流水线、切块策略、嵌入模型、Hybrid Search、重排、知识图谱增强 |
| [06_Evaluation.md](./06_Evaluation.md) | 模型评估 | 传统NLP指标局限、基准测试(MMLU/GSM8K等)、LLM-as-Judge、Agent评估、红队测试 |
| [07_LLM_Future.md](./07_LLM_Future.md) | LLM前沿展望 | AGI差距、多模态融合、世界模型、合成数据、具身智能、新架构(Mamba)、应用预测 |
| [08_AIGC.md](./08_AIGC.md) | AIGC生成式AI | 扩散模型原理、文生图架构(UNet/DiT)、物理规律问题、多模态理解、质量评估(FID/CLIP Score)、指令理解优化、美感建模、模糊意图处理、主流方案对比、视频生成 |
| [09_Infra.md](./09_Infra.md) | 训练基础设施 | 训练加速总览、DeepSpeed详解、多机多卡分布式训练、训练效率优化、收敛速度优化、通信问题、梯度累加、GPU推理优化、量化方法(GPTQ/AWQ/FP8)、Loss震荡分析 |

---

## 📖 各文档内容概览

### 01_LLM_Basics.md
涵盖大模型核心技术和面试高频考点：自注意力机制、位置编码(绝对/相对/ROPE)、MHA/MQA/GQA对比、LLM架构对比、Scaling Laws、解码策略、词元化算法、涌现能力、激活函数、MoE原理、分布式训练(3D并行/ZeRO/FSDP)、长上下文处理、FlashAttention、KV Cache、vLLM、LoRA/QLoRA、AdamW优化器、Self-Attention vs Cross-Attention、Encoding方式全览(BPE/BBPE/WordPiece/Unigram)、词表生成与使用、Token向量化计算、多模态Transformer架构设计(投影/交叉注意力/统一)、GLM模型技术特点与架构、Prompt工程与迭代优化、意图识别。

### 02_VLM.md
聚焦视觉-语言多模态模型：模态对齐与融合方法、CLIP对比学习、LLaVA/MiniGPT-4架构、视觉指令微调、视频时序建模、Grounding视觉定位、高分辨率图像处理、VLM幻觉问题及缓解、VLA(视觉-语言-动作)模型与具身智能、图像与文本Embedding相似度衡量、多模态+GRPO结合思路。

### 03_RLHF.md
详解强化学习对齐技术：RLHF vs SFT对比、三阶段流程、奖励模型设计、PPO算法详解、KL惩罚系数调节、奖励黑客问题、DPO/GRPO/DAPO/GSPO等优化算法、模式化问题、信用分配机制、RLAIF、贝尔曼方程、SFT数据构建全流程、为什么RL前需要SFT、PPO vs GRPO vs DPO全面对比、GRPO优势值评估机制、Reward Model训练详解、RM与RLVR关系、CoT思维链必要性与质量评估、持续学习与RL未来发展、Human-in-the-loop模型标注问题、RL算法认知与落地。

### 04_Agent.md
围绕LLM Agent构建：核心组件(规划/记忆/行动)、ReAct框架、CoT/ToT/GoT规划能力、记忆系统设计、Tool Use/Function Calling、LangChain vs LlamaIndex对比、多智能体系统、具身Agent vs 软件Agent、Agent安全对齐、A2A协议、Agent微调、MCP协议、任务拆分策略、长上下文处理、模型能力vs框架设计、记忆方案扩展(MemGPT/MemoryBank/Reflexion)、Multi-Agent协作模式(Swarm)、Agent微调vs模型微调。

### 05_RAG.md
系统讲解检索增强生成：RAG原理与优势对比、离线+在线流水线、文本切块策略、嵌入模型选择与评估、Hybrid Search/Rerank/HyDE等检索优化、Lost in Middle问题、RAGAS评估、知识图谱增强、Adaptive/Iterative/Self-RAG等高级范式。

### 06_Evaluation.md
聚焦模型与Agent评估：传统BLEU/ROUGE局限性、主流基准测试(MMLU/GSM8K/HumanEval等)、LLM-as-Judge方法与偏见缓解、事实性/推理/安全性评估、Agent评估难点与基准、WebArena/SWE-bench等、红队测试方法、人工评估设计、部署后持续监控。

### 07_LLM_Future.md
展望LLM未来发展：AGI关键缺失能力(因果推理/长程规划/世界模型)、多模态融合趋势、合成数据与模型坍缩、具身智能与机器人、个性化与隐私平衡、Transformer vs Mamba等新架构、3-5年颠覆性应用预测。

### 08_AIGC.md
系统讲解AIGC生成式AI：扩散模型原理(DDPM/DDIM/Score-based)、文生图模型架构(VAE+TextEncoder+UNet/DiT)、生图物理规律问题与ControlNet、多模态理解对AIGC的作用、绘图质量评估(FID/CLIP Score/IS/Aesthetic Score)、指令理解优化策略、美感建模与风格迁移、模糊意图处理、主流方案对比(SD/SDXL/SD3/Flux/DALL-E3/Midjourney)、Flow Matching、文生图→图生图→图生视频(Sora/SVD/AnimateDiff)。

### 09_Infra.md
聚焦训练基础设施与加速：训练加速总览(计算/通信/显存/数据)、DeepSpeed ZeRO 1/2/3详解与Offload、多机多卡分布式训练(DDP/FSDP/Ring AllReduce/NCCL)、训练效率优化(MFU/HFU/Checkpoint)、收敛速度优化(学习率调度/梯度裁剪)、多机多卡通信问题(带宽/延迟/拓扑)、梯度累加原理与数学等价性、GPU训练/推理性能优化(vLLM/TensorRT-LLM/SGLang/Speculative Decoding)、量化方法(GPTQ/AWQ/KV Cache量化/FP8训练)、Loss震荡原因分析与诊断。
