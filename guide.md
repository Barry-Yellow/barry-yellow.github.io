# Project Guide — barry-yellow.github.io

> AI 快速上手文档。新 session 读此文件即可了解整个项目结构，无需重新扫描所有文件。
> 最后更新：2026-04-21

---

## 一、项目概况

| 项目 | 值 |
|------|-----|
| 框架 | **Docsify**（GitHub Pages 托管，纯静态，无构建步骤） |
| 仓库 | `barry-yellow.github.io`，部署在 `main` 分支 |
| 内容语言 | 中文为主，技术术语英文 |
| 写作风格 | 学术笔记，含 LaTeX 公式、架构图、实验结果 |
| Markdown 文件数 | ~100 篇 |
| 主题文件夹 | 10 个（见下） |

---

## 二、配置文件（不要改动）

| 文件 | 作用 |
|------|------|
| `index.html` | Docsify 主配置：插件、主题、KaTeX LaTeX 宏、全文搜索 |
| `_coverpage.md` | 封面页（Barry's Notes + 名言 + GitHub 链接） |
| `_navbar.md` | 顶部导航栏（Markdown 指南、相关链接） |
| `_sidebar.md` | 全局侧边栏（10 个主题文件夹入口） |
| `_config.yml` | Jekyll 配置（GitHub Pages 用，实际由 `.nojekyll` 禁用 Jekyll） |
| `.nojekyll` | 禁用 Jekyll，让 GitHub Pages 直接托管静态文件 |
| `my_style.css` | 自定义样式 |
| `back_to_top.css` / `back_to_top.js` | 回到顶部按钮 |
| `assets/` | 主题相关 CSS/JS/SASS（不要修改） |
| `mpe_guide.md` | Markdown Preview Enhanced 指南（独立文件，不在导航中） |

---

## 三、Docsify 侧边栏机制

- 每个文件夹有自己的 `_sidebar.md`，Docsify 会自动加载当前路径下最近的 `_sidebar.md`
- 链接路径**从项目根目录开始**，如 `MAS/Topology/DyLAN.md`（不加前导 `/`）
- 全局 `_sidebar.md` 列出 10 个主题入口，各子文件夹的 `_sidebar.md` 列出该主题的文章
- 子文件夹 sidebar 末尾通常有 `[Back to Homepage](README.md)` 或 `[Back to MAS](MAS/README.md)`

---

## 四、文件夹结构与内容

### 4.1 MAS/ — 多智能体系统（LLM-based）

**`MAS/_sidebar.md`** 结构：
- 拓扑优化 → Topology/
- 提示词优化 → Prompt/
- 失败分析 → FailAndWhy/
- 记忆机制 → Memory/

| 子文件夹 | 文件 | 说明 |
|----------|------|------|
| `MAS/` 根 | `README.md` | 概述 |
| | `references.md` | 参考文献列表 |
| `MAS/Topology/` | `MASFrameWork.md` | MAS 框架对比分析（新增，已有 .html 版） |
| | `DyLAN.md` | DyLAN 论文笔记（内容极少，待补充） |
| | `AFlow.md` | AFlow 论文笔记（内容极少，待补充） |
| `MAS/Prompt/` | `MAS_GPT.md` | 训练 LLM 用于多智能体系统 ✅ |
| | `MIPRO.md` | MIPRO 提示词优化（内容极少，待补充） |
| `MAS/FailAndWhy/` | `WhyFail.md` | 多智能体为何失败 |
| | `WhoAndWhen.md` | 失败归因方法 |
| | `AgentFixer.md` | 从失败检测到修复建议 |
| | `ClaudeMechanism.md` | Claude 的失效处理机制 |
| `MAS/Memory/` | `MemoryIntro.md` | 记忆机制概述 |

**注意：** `MAS/` 下有 `ClaudeCode架构深度解析.html` 和 `MAS/Topology/MASFrameWork.html`，是独立 HTML 展示文件，不在 sidebar 中。

---

### 4.2 MARL/ — 多智能体强化学习

**`MARL/_sidebar.md`**：MARL 基础、MARL 安全综述 2023

| 文件 | 说明 |
|------|------|
| `marl_basics.md` | MARL 基础概念 |
| `survey2023.md` | 安全 MARL 综述 2023 |

---

### 4.3 LM/ — 语言模型

**`LM/_sidebar.md`**：BERT、NLP基础、大模型基础、强化学习基础、Transformer、MoE

| 文件 | 说明 |
|------|------|
| `Transformer.md` | Transformer 架构详解 ✅ |
| `NLP.md` | NLP 基础：词嵌入、ELMo ✅ |
| `RL_base.md` | REINFORCE、Actor-Critic、PPO ✅ |
| `LMIntro.md` | Prompt Engineering、RAG、微调概述 |
| `BERT.md` | 内容极少（只有参考链接），待补充 |
| `MoE.md` | 完全空白，待补充 |

---

### 4.4 DMT/ — 深度多模态学习

**`DMT/_sidebar.md`**：多模态概念笔记、导论、CLIP、BLIP、BLIP2、Diffusion、自回归、NLP基础、VidKV量化

| 文件 | 说明 |
|------|------|
| `CLIP.md` | CLIP 模型详解 ✅ |
| `BLIP.md` | BLIP 模型详解 ✅ |
| `BLIP2.md` | BLIP-2 Q-Former 详解 ✅ |
| `Diffusion.md` | 扩散模型原理 ✅ |
| `AutoAggressive.md` | 自回归 vs 扩散对比 ✅ |
| `DMTIntro.md` | 多模态学习导论（内容散乱，待优化） |
| `schedule.md` | 多模态概念笔记（标题误导，实为概念笔记） |
| `NLPLearning.md` | 完全空白，待补充或删除 |
| `VidKV.md` | 视频 LLM KV Cache 量化（多模态视角） ✅ |

---

### 4.5 SafeRL/ — 安全强化学习

**`SafeRL/_sidebar.md`**：survey 2024、survey 2023、schedule、Safe Critic、More 2024

| 文件 | 说明 |
|------|------|
| `survey2024.md` | IJCAI 2024 SafeRL 综述 ✅ |
| `survey2023.md` | 基于状态的 SafeRL 综述 ✅ |
| `more2024.md` | 约束公式化的 8 种类型 ✅ |
| `schedule.md` | 研究进度记录（混合了进度和笔记） |
| `safeCritic.md` | 完全空白，待补充或删除 |

---

### 4.6 Mamba/ — Mamba 架构（SSM）

**`Mamba/_sidebar.md`**：Mamba 基础、Mamba+频率分解实验、Mamba 改进

| 文件 | 说明 |
|------|------|
| `mamba_theory.md` | SSM 基础、Mamba 架构原理 ✅ |
| `mamba_improve.md` | Mamba 改进实验（结果不稳定，分析不完整） |
| `mamba_freq_result.md` | 频率分解实验结果（大量空表格，待补充） |

---

### 4.7 Frequency/ — 频域分析与 Transformer 预测

**`Frequency/_sidebar.md`** 结构：
- 频域基础：Cross_cov、EMD_X11_diff
- 频域 Transformer：freq_transformer、SpectFormer、AutoFormer、FEDformer、DCTTransformer
- Transformer 时序预测：transformer_time_series_prediction、Informer_PatchTST
- 跨文件夹链接：asset_pricing/different_views...

| 文件 | 说明 |
|------|------|
| `Cross_cov.md` | 自相关系数、自谱密度、互协方差 ✅ |
| `EMD_X11_diff.md` | EMD、X11、差分等平稳方法 ✅ |
| `freq_transformer.md` | 频率+Transformer 综述（CV方向） ✅ |
| `SpectFormer_freq_att.md` | SpectFormer 频率注意力 ✅ |
| `AutoFormer.md` | AutoFormer 自相关机制 ✅ |
| `FEDformer.md` | FEDformer 频率增强 ✅ |
| `DCTTransformer.md` | FECAM：DCT 改进 SE 架构 ✅ |
| `Informer_PatchTST.md` | Informer 与 PatchTST ✅ |
| `transformer_time_series_prediction.md` | 时间序列预测经典模型 ✅ |

---

### 4.8 Time_Series_Prediction/ — 时间序列预测

**`Time_Series_Prediction/_sidebar.md`**：LSTM+Frequency、小波+注意力、多变量LSTM、小波+投资组合、跨文件夹链接

| 文件 | 说明 |
|------|------|
| `Multi-Frequency_Trading_Patterns_LSTM.md` | 多频率交易模式 + LSTM ✅ |
| `wavelet_attention_lstm.md` | 小波变换 + 注意力 + LSTM ✅ |
| `multi-input_attention.md` | 多变量 LSTM + 注意力 ✅ |
| `WISE_wavelet_portfolio.md` | 小波变换 + 投资组合 ✅ |

---

### 4.9 asset_pricing/ — 资产定价

**`asset_pricing/_sidebar.md`**：SDF与回报空间、风险价格与风险溢价、SDF/β/均值方差联系、HJ边界、特征预处理

| 文件 | 说明 |
|------|------|
| `sdf_and_payoff_space.md` | SDF 与或有权益、状态资产定价 ✅ |
| `prices_of_risk_and_risk_premia.md` | 风险价格与风险溢价的区分 ✅ |
| `relations_between_different_frameworks.md` | SDF、β 表达式、均值方差前沿的联系 ✅ |
| `hj_bound_and_hj_distance.md` | HJ 边界与 HJ 距离 ✅ |
| `different_views_of_charac_and_data_preprocessing.md` | Barra 模型、特征标准化（主版本，其他 sidebar 跨文件夹引用此文件） ✅ |

---

### 4.10 Quantization/ — 模型量化

**`Quantization/_sidebar.md`**：VidKV、旋转量化、量化基础、QuantCode、KIVI、QuaRot、Lexico、calibQuant

| 文件 | 说明 |
|------|------|
| `QuantIntro.md` | 量化基础：PTQ、QAT、量化公式 ✅ |
| `Rotation_Quantization.md` | 旋转量化方法 ✅ |
| `VidKV.md` | KV Cache 量化（量化视角，与 DMT/VidKV.md 内容不同） ✅ |
| `QuantCode.md` | 量化代码实现 |
| `KIVI.md` | KIVI 量化方法 ✅ |
| `QuaRot.md` | QuaRot 旋转量化 ✅ |
| `Lexico.md` | Lexico 稀疏编码压缩 ✅ |
| `calibQuant.md` | CalibQuant 多模态量化 ✅ |

---

### 4.11 temp/ — 未归档暂存区

存放尚未归档的草稿和参考资料，内容确认后应移入对应主题文件夹。当前为空。

---

## 五、已知问题清单

详细问题记录见 `TotalPage.md`（项目全局索引与改写参考手册）。

### 空文件（需补充内容或删除）
- `LM/BERT.md` — 只有参考链接
- `LM/MoE.md` — 完全空白
- `DMT/NLPLearning.md` — 完全空白
- `SafeRL/safeCritic.md` — 完全空白
- `MAS/Topology/DyLAN.md` — 内容极少
- `MAS/Topology/AFlow.md` — 内容极少
- `MAS/Prompt/MIPRO.md` — 内容极少

### 跨文件夹引用（sidebar 中的跨目录链接）
以下 sidebar 引用了其他文件夹的文件，这是刻意设计的（避免重复）：
- `Frequency/_sidebar.md` → `asset_pricing/different_views_of_charac_and_data_preprocessing.md`
- `Time_Series_Prediction/_sidebar.md` → `asset_pricing/sdf_and_payoff_space.md` 和 `asset_pricing/different_views...`
- `DMT/_sidebar.md` → `Quantization/VidKV.md`

---

## 六、新增文章操作流程

1. 在对应文件夹创建 `.md` 文件
2. 在该文件夹的 `_sidebar.md` 添加条目（路径从根目录开始，不加 `/`）
3. 如果是新文件夹，需在根目录 `_sidebar.md` 和 `README.md` 添加入口
4. 图片放在对应文件夹的 `image/` 子目录，路径写相对路径

## 七、LaTeX 公式

- 行内公式：`$...$`
- 块级公式：`$$...$$`
- 由 `index.html` 中的 KaTeX 插件渲染，支持常见数学符号

## 八、TotalPage.md 说明

`TotalPage.md` 是项目的**改写工作参考手册**，记录了每个文件的质量评估、问题清单和改写建议。AI 执行批量改写任务时应优先读取该文件。
