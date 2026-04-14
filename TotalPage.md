# 项目全局索引与改写参考手册

> 本文件是改写工作的参考文档，记录了每个文件夹、每个文件的当前状态、问题、和预期目标。
> 改写时逐项对照，完成一个勾一个。

---

## 一、项目概况

| 项目 | 值 |
|------|-----|
| 框架 | docsify（GitHub Pages 托管） |
| 总大小 | ~148MB（含图片） |
| Markdown 文件数 | 102 个（不含 MAS/aa 副本） |
| 图片文件数 | ~80+ 张（分散在各文件夹） |
| 主题文件夹 | 12 个 |
| 内容语言 | 中文为主，技术术语英文 |
| 写作风格 | 学术笔记，含公式推导、架构图、实验结果 |

---

## 二、配置文件（不要改动）

| 文件 | 说明 |
|------|------|
| `index.html` | docsify 主配置，含插件、主题、LaTeX 宏 |
| `_coverpage.md` | 封面页 |
| `_navbar.md` | 顶部导航栏 |
| `_config.yml` | Jekyll 配置（GitHub Pages 用） |
| `.nojekyll` | 禁用 Jekyll |
| `my_style.css` | 自定义样式 |
| `back_to_top.css` / `back_to_top.js` | 回到顶部按钮 |
| `assets/` | 主题相关 CSS/JS/SASS |
| `mpe_guide.md` | Markdown Preview Enhanced 指南（独立文件，不在导航中） |

---

## 三、全局问题清单

### 3.1 需要删除的内容
| 项目 | 原因 |
|------|------|
| `MAS/aa/` 整个目录 | MAS/ 的完整副本，未被任何导航引用 |
| `Frequency_Transformer/GFNet.py` | Python 代码文件，不属于文档站 |
| `Mamba/todo.md` (5行) | 待办清单，无文档价值 |
| `Frequency_Transformer/todo.md` (5行) | 待办清单，无文档价值 |
| `DMT/todo.md` (2行) | 待办清单，无文档价值 |
| `MAS/todo.md` (81行) | 研究大纲/待办，可并入参考文献或删除 |
| `MAS/bib.md` (655行) | 纯 BibTeX 条目，考虑合并到 references.md 或删除 |

### 3.2 确认的重复文件
| 文件A | 文件B | 关系 |
|-------|-------|------|
| `Frequency_Transformer/different_views_of_charac_and_data_preprocessing.md` | `asset_pricing/different_views_of_charac_and_data_preprocessing.md` | **完全相同**（MD5一致） |
| `Frequency/SpectFormer_freq_att.md` | `Frequency_Transformer/SpectFormer_freq_att.md` | **完全相同**（MD5一致） |
| `Time_Series_Prediction/different_views_of_charac_and_data_preprocessing.md` | 上面两个 | **内容不同**，各自独立 |
| `Transformer_Prediction/different_views_of_charac_and_data_preprocessing.md` | 上面的 | **内容不同**，各自独立 |
| `Time_Series_Prediction/sdf_and_payoff_space.md` | `asset_pricing/sdf_and_payoff_space.md` | **内容不同**，各自独立 |
| `DMT/VidKV.md` | `Quantization/VidKV.md` | **内容不同**，各自独立 |
| `MARL/survey2023.md` | `SafeRL/survey2023.md` | **内容不同**，各自独立 |

**处理建议：** 完全相同的两组，只在最合适的文件夹保留一份，其他 sidebar 改为跨文件夹链接。

### 3.3 Sidebar 指向不存在的文件（断链）
| Sidebar 文件 | 断链条目 | 目标文件 |
|-------------|----------|----------|
| `MAS/Topology/_sidebar.md` | MaaS | `MAS/Topology/MaaS.md` ❌ |
| `MAS/Prompt/_sidebar.md` | EvoPrompt | `MAS/Prompt/EvoPrompt.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | WhyFail | `MAS/FailAndWhy/WhyFail.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | WhoAndWhen | `MAS/FailAndWhy/WhoAndWhen.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | whereAndHow | `MAS/FailAndWhy/whereAndHow.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | AgentTracer | `MAS/FailAndWhy/AgentTracer.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | TRAIL | `MAS/FailAndWhy/TRAIL.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | SHIELDA | `MAS/FailAndWhy/SHIELDA.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | GraphTracer | `MAS/FailAndWhy/GraphTracer.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | Aegis | `MAS/FailAndWhy/Aegis.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | AgentAsk | `MAS/FailAndWhy/AgentAsk.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | RAFFLES | `MAS/FailAndWhy/RAFFLES.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | CanAgentsFixAgentIssues | `MAS/FailAndWhy/CanAgentsFixAgentIssues.md` ❌ |
| `MAS/FailAndWhy/_sidebar.md` | HallucinationsSurvey | `MAS/FailAndWhy/HallucinationsSurvey.md` ❌ |

**共 14 个断链**，全部在 MAS 子目录。

### 3.4 内容错误
| 文件 | 问题 |
|------|------|
| `MAS/Prompt/README.md` | 标题写"拓扑优化"，但内容是"提示词优化" |
| `Quantization/_sidebar.md` | 两个条目都叫"量化知识"，无法区分 |
| `MARL/_sidebar.md` | 条目名写"SafeRL survey IJCAI 2023"，应该是 MARL 相关 |
| `DMT/_sidebar.md` | 列了 NLPLearning.md 但该文件为空 |

---

## 四、逐文件夹详细索引

### 4.1 asset_pricing/ — 资产定价

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 264KB |
| 文章数 | 5 篇 + README |
| 图片 | 1 张（image/） |
| 整体质量 | ⭐⭐⭐⭐ 高质量，数学推导完整 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 84B | 空壳 | 一行标题 + sidebar include | 无概述 | 写 200-400 字概述：SDF 框架、风险溢价、HJ 边界等内容的关系和学习路径 |
| `sdf_and_payoff_space.md` | 178行 | 7.5KB | ✅好 | SDF 与或有权益、状态资产定价 | 无明显问题 | 检查可读性，删冗余句 |
| `prices_of_risk_and_risk_premia.md` | 252行 | 13.6KB | ✅好 | 风险价格与风险溢价的区分 | 无明显问题 | 检查可读性 |
| `relations_between_different_frameworks.md` | 262行 | 8.5KB | ✅好 | SDF、β 表达式、均值方差前沿的联系 | 无明显问题 | 检查可读性 |
| `hj_bound_and_hj_distance.md` | 55行 | 4.0KB | ✅好 | HJ 边界与 HJ 距离 | 无明显问题 | 检查可读性 |
| `different_views_of_charac_and_data_preprocessing.md` | 109行 | 6.9KB | ✅好 | Barra 模型、特征标准化 | 与 Frequency_Transformer 版完全相同 | 保留此份作为主版本 |

**Sidebar 当前状态：** 5 个条目 + Back to Homepage，链接均有效 ✅

---

### 4.2 Time_Series_Prediction/ — 时间序列预测

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 2.6MB |
| 文章数 | 7 篇 + README |
| 图片 | ~40 张（image/） |
| 整体质量 | ⭐⭐⭐⭐ 多数文章质量好 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 102B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述：LSTM+频率、小波变换、注意力机制等方向 |
| `Multi-Frequency_Trading_Patterns_LSTM.md` | 162行 | 10.5KB | ✅好 | 多频率交易模式 + LSTM | 图片路径有反斜杠 `\` | 修正路径为 `/` |
| `wavelet_attention_lstm.md` | 43行 | 3.0KB | ✅好 | 小波变换 + 注意力 + LSTM | 无明显问题 | 检查可读性 |
| `multi-input_attention.md` | 27行 | 3.5KB | ✅好 | 多变量 LSTM + 注意力 | 无明显问题 | 检查可读性 |
| `thinking_new_model.md` | 7行 | 359B | ❌空 | 只有图片引用，无文字 | 几乎无内容 | 要么补充思考过程文字，要么删除并从 sidebar 移除 |
| `WISE_wavelet_portfolio.md` | 219行 | 12.5KB | ✅好 | 小波变换 + 投资组合 | 无明显问题 | 检查可读性 |
| `sdf_and_payoff_space.md` | 178行 | 7.5KB | ✅好 | SDF 与回报空间 | 与 asset_pricing 版内容不同 | 保留，但考虑是否属于此文件夹 |
| `different_views_of_charac_and_data_preprocessing.md` | 109行 | 6.9KB | ✅好 | 特征视角与预处理 | 与 asset_pricing 版内容不同 | 保留 |

**Sidebar 当前状态：** 7 个条目 + Back to Homepage，链接均有效 ✅

---

### 4.3 Frequency/ — 频域分析

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 860KB |
| 文章数 | 3 篇 + README |
| 图片 | 1 张（image/） |
| 整体质量 | ⭐⭐⭐ 文章质量好但数量少 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 81B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述：频域基础知识体系 |
| `Cross_cov.md` | 122行 | 8.9KB | ✅好 | 自相关系数、自谱密度、互协方差 | 无明显问题 | 检查可读性 |
| `EMD_X11_diff.md` | 99行 | 7.5KB | ✅好 | EMD、X11、差分等平稳方法 | 无明显问题 | 检查可读性 |
| `SpectFormer_freq_att.md` | 71行 | 8.4KB | ✅好 | SpectFormer 频率注意力 | 与 Frequency_Transformer 版**完全相同** | 删除此份，sidebar 改为链接到 Frequency_Transformer 版 |

**Sidebar 当前状态：** 3 个条目 + Back to Homepage ✅

---

### 4.4 Frequency_Transformer/ — 频域 Transformer

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 3.3MB |
| 文章数 | 6 篇 + README（不含 todo 和 GFNet.py） |
| 图片 | 16 张（image/） |
| 整体质量 | ⭐⭐⭐⭐ 文章质量好 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 102B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述 |
| `freq_transformer.md` | 33行 | 1.9KB | ✅好 | 频率+Transformer 综述（FNet、GFNet、SpectFormer） | 无明显问题 | 检查可读性 |
| `SpectFormer_freq_att.md` | 71行 | 8.4KB | ✅好 | SpectFormer 详解 | 与 Frequency 版完全相同 | 保留此份作为主版本 |
| `AutoFormer.md` | 60行 | 6.8KB | ✅好 | AutoFormer 自相关机制 | 无明显问题 | 检查可读性 |
| `FEDformer.md` | 71行 | 8.4KB | ✅好 | FEDformer 频率增强 | 无明显问题 | 检查可读性 |
| `DCTTransformer.md` | 55行 | 5.6KB | ✅好 | FECAM：DCT 改进 SE 架构 | 无明显问题 | 检查可读性 |
| `different_views_of_charac_and_data_preprocessing.md` | 109行 | 6.9KB | ✅好 | 特征视角与预处理 | 与 asset_pricing 版**完全相同** | 删除此份，sidebar 改为链接到 asset_pricing 版 |
| `todo.md` | 5行 | 176B | 待办 | 待办清单 | 无文档价值 | 删除 |
| `GFNet.py` | — | — | 代码 | Python 代码文件 | 不属于文档站 | 删除 |

**Sidebar 当前状态：** 6 个条目 + Back to Homepage ✅（但 different_views 重复需处理）

---

### 4.5 Transformer_Prediction/ — Transformer 预测

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 24KB |
| 文章数 | 2 篇 + README |
| 图片 | 无 |
| 整体质量 | ⭐⭐ 内容薄弱，需要补充 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 120B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述 |
| `transformer_time_series_prediction.md` | 117行 | 7.0KB | ✅好 | Transformer 时间序列预测经典模型 | 无明显问题 | 检查可读性 |
| `different_views_of_charac_and_data_preprocessing.md` | 108行 | 6.9KB | ✅好 | 特征视角与预处理 | 与其他版本内容不同 | 保留 |

**Sidebar 当前状态：** 2 个条目 + Back to Homepage ✅
**补充建议：** 文件夹内容太少，建议补充 1-2 篇（如 Informer、PatchTST 等经典模型笔记）

---

### 4.6 Mamba/ — Mamba 架构

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 2.2MB |
| 文章数 | 3 篇 + README（不含 todo） |
| 图片 | 18 张（image/） |
| 整体质量 | ⭐⭐⭐ 理论篇好，实验篇粗糙 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 68B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述：SSM 基础、Mamba 原理、频率实验 |
| `mamba_theory.md` | 65行 | 7.0KB | ✅好 | SSM 基础、Mamba 架构原理 | 作者标注公式排版未完成 | 优化公式排版，补全未完成部分 |
| `mamba_improve.md` | 41行 | 4.5KB | ⚠️中 | Mamba 改进实验 | 结果不稳定，分析不完整 | 优化结构，补充分析 |
| `mamba_freq_result.md` | 54行 | 1.1KB | ⚠️粗糙 | 频率分解实验结果 | 大量空表格，缺少解释 | 补充实验说明和结果分析 |
| `todo.md` | 5行 | 176B | 待办 | 待办清单 | 无文档价值 | 删除 |

**Sidebar 当前状态：** 3 个条目 + Back to Homepage ✅

---

### 4.7 LM/ — 语言模型

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 2.9MB |
| 文章数 | 6 篇 + README |
| 图片 | 17 张（LM/ 根目录下 image-*.png） |
| 整体质量 | ⭐⭐⭐ 部分好，部分空壳 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 80B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述：NLP 基础、Transformer、BERT、RL、MoE |
| `Transformer.md` | 48行 | 3.5KB | ✅好 | Transformer 架构详解 | 无明显问题 | 检查可读性 |
| `NLP.md` | 40行 | 2.7KB | ✅好 | NLP 基础：词嵌入、ELMo | 无明显问题 | 检查可读性 |
| `RL_base.md` | 254行 | 15.7KB | ✅好 | REINFORCE、Actor-Critic、PPO | 无明显问题 | 检查可读性 |
| `LMIntro.md` | 27行 | 1.4KB | ⚠️中 | Prompt Engineering、RAG、微调概述 | 内容散乱，深度不够 | 重写为更系统的大模型入门 |
| `BERT.md` | 6行 | 581B | ❌空 | 只有参考链接，无正文 | 无实质内容 | 写 BERT 笔记（MLM、NSP、微调） |
| `MoE.md` | 4行 | 4B | ❌空 | 完全空白 | 无内容 | 写 MoE 笔记（门控机制、稀疏激活、GShard/Switch Transformer） |

**Sidebar 当前状态：** 6 个条目，无 Back to Homepage（需补上）✅

---

### 4.8 SafeRL/ — 安全强化学习

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 1.5MB |
| 文章数 | 5 篇 + README |
| 图片 | 4 张（SafeRL/ 根目录下） |
| 整体质量 | ⭐⭐⭐⭐ 综述类文章质量高 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 121B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述 |
| `survey2024.md` | 33行 | 5.8KB | ✅好 | IJCAI 2024 SafeRL 综述 | 无明显问题 | 检查可读性 |
| `survey2023.md` | 117行 | 32.7KB | ✅好 | 基于状态的 SafeRL 综述 | 无明显问题 | 检查可读性 |
| `schedule.md` | 211行 | 13.4KB | ⚠️中 | 研究进度记录 | 混合了进度和笔记 | 提取有价值的笔记内容，删除纯进度记录 |
| `safeCritic.md` | 0行 | 0B | ❌空 | 完全空白 | 无内容 | 写 Safe Critic 笔记或删除并从 sidebar 移除 |
| `more2024.md` | 42行 | 20.6KB | ✅好 | 约束公式化的 8 种类型 | 无明显问题 | 检查可读性 |

**Sidebar 当前状态：** 5 个条目，无 Back to Homepage（需补上）✅

---

### 4.9 MARL/ — 多智能体强化学习

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 40KB |
| 文章数 | 1 篇 + README |
| 图片 | 无 |
| 整体质量 | ⭐⭐ 内容极度薄弱 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 103B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述 |
| `survey2023.md` | 114行 | 32.5KB | ✅好 | 基于状态的安全 MARL 综述 | sidebar 条目名写的是"SafeRL survey" | 修正 sidebar 条目名 |

**Sidebar 当前状态：** 1 个条目，条目名错误（写成 SafeRL），无 Back to Homepage ✅
**补充建议：** 文件夹内容极少，建议补充 1-2 篇（MARL 基础概念、QMIX/MAPPO 等经典算法）

---

### 4.10 DMT/ — 深度多模态学习

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 14MB（图片占大头） |
| 文章数 | 8 篇 + README（不含 todo） |
| 图片 | 21 张（DMT/ 根目录下 image-*.png） |
| 整体质量 | ⭐⭐⭐⭐ CLIP/BLIP 系列质量高，部分文件空 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 110B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述：多模态学习路线（对比学习→CLIP→BLIP→扩散→自回归） |
| `DMTIntro.md` | 20行 | 2.3KB | ⚠️粗糙 | SimCLR、MOCO、FID、CLIP score 零散笔记 | 无结构，像随手记 | 重写为多模态学习系统性介绍 |
| `CLIP.md` | 118行 | 12.0KB | ✅好 | CLIP 模型详解 | 无明显问题 | 检查可读性 |
| `BLIP.md` | 133行 | 13.4KB | ✅好 | BLIP 模型详解 | 无明显问题 | 检查可读性 |
| `BLIP2.md` | 84行 | 4.3KB | ✅好 | BLIP-2 Q-Former 详解 | 无明显问题 | 检查可读性 |
| `Diffusion.md` | 229行 | 13.1KB | ✅好 | 扩散模型原理 | 无明显问题 | 检查可读性 |
| `AutoAggressive.md` | 97行 | 7.0KB | ✅好 | 自回归 vs 扩散对比 | 无明显问题 | 检查可读性 |
| `NLPLearning.md` | 0行 | 0B | ❌空 | 完全空白 | 无内容 | 写 NLP 学习笔记或删除并从 sidebar 移除 |
| `VidKV.md` | 121行 | 9.5KB | ✅好 | 视频 LLM KV Cache 量化 | 与 Quantization 版内容不同 | 保留，侧重多模态视角 |
| `schedule.md` | 68行 | 5.3KB | ⚠️中 | 多模态概念笔记 | 标题叫 schedule 但内容是概念笔记，结构散乱 | 重命名或重组内容 |
| `todo.md` | 2行 | 52B | 待办 | 待办清单 | 无文档价值 | 删除 |

**Sidebar 当前状态：** 9 个条目 + Back to Homepage ✅（但 NLPLearning 指向空文件）

---

### 4.11 Quantization/ — 模型量化

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 72KB |
| 文章数 | 8 篇 + README |
| 图片 | 无 |
| 整体质量 | ⭐⭐⭐ 多数文章好，个别空壳 |

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 89B | 空壳 | 一行英文标题 | 无概述 | 写 200-400 字概述 |
| `QuantIntro.md` | 33行 | 3.0KB | ✅好 | 量化基础：PTQ、QAT、量化公式 | sidebar 条目名"量化知识"与另一个重复 | 改 sidebar 名为"量化基础" |
| `Rotation_Quantization.md` | 92行 | 6.7KB | ⚠️注意 | 旋转量化方法 | sidebar 条目名"量化知识"重复；首行内容需确认 | 改 sidebar 名为"旋转量化" |
| `VidKV.md` | 169行 | 14.5KB | ✅好 | KV Cache 量化（量化视角） | 与 DMT 版内容不同 | 保留 |
| `QuantCode.md` | 99行 | 4.4KB | ⚠️中 | 量化代码实现 | 笔记风格，缺乏深度 | 优化结构 |
| `KIVI.md` | 38行 | 1.6KB | ✅好 | KIVI 量化方法 | 无明显问题 | 检查可读性 |
| `QuaRot.md` | 74行 | 5.0KB | ✅好 | QuaRot 旋转量化 | 无明显问题 | 检查可读性 |
| `Lexico.md` | 74行 | 6.7KB | ✅好 | Lexico 稀疏编码压缩 | 无明显问题 | 检查可读性 |
| `calibQuant.md` | 60行 | 5.8KB | ✅好 | CalibQuant 多模态量化 | 无明显问题 | 检查可读性 |

**Sidebar 当前状态：** 8 个条目 + Back to Homepage，两个"量化知识"需改名 ⚠️

---

### 4.12 MAS/ — 多智能体系统

| 属性 | 值 |
|------|-----|
| 文件夹大小 | 1.1MB（含 aa 副本） |
| 子文件夹 | 4 个（Topology、Prompt、Memory、FailAndWhy）+ aa 副本 |
| 整体质量 | ⭐⭐ 问题最多的文件夹 |

**根目录文件：**

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 7行 | 173B | ⚠️ | 一行标题 + 腾讯文档外链 | 外链不适合文档站 | 去掉外链，写 200-400 字概述 |
| `bib.md` | 655行 | 29.6KB | 参考 | 纯 BibTeX 条目 | 体量大但无阅读价值 | 合并有用部分到 references.md，删除此文件 |
| `references.md` | 95行 | 4.0KB | ✅好 | 参考文献列表 | 无明显问题 | 保留，可补充 bib.md 中有用的条目 |
| `todo.md` | 81行 | 6.9KB | 待办 | 研究大纲/待办 | 无文档价值 | 删除 |

**MAS/Topology/ — 拓扑优化：**

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 110B | 空壳 | 标题 + 一句话 | 无概述 | 补充概述 |
| `DyLAN.md` | 0行 | 3B | ❌空 | 只有"gseg" | 无内容 | 写 DyLAN 论文笔记 |
| `AFlow.md` | 0行 | 4B | ❌空 | 只有"gseg" | 无内容 | 写 AFlow 论文笔记 |
| `_sidebar.md` 断链 | — | — | — | 引用 MaaS.md | 文件不存在 | 创建 MaaS.md 或从 sidebar 移除 |

**MAS/Prompt/ — 提示词优化：**

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 113B | ❌错误 | 标题写"拓扑优化" | 标题与内容不符 | 改标题为"提示词优化" |
| `MIPRO.md` | 58行 | 2.6KB | ❌空 | 只有一句"总结全文" | 无实质内容 | 写 MIPRO 论文笔记 |
| `MAS_GPT.md` | 82行 | 7.5KB | ✅好 | 训练 LLM 用于多智能体系统 | 无明显问题 | 检查可读性 |
| `_sidebar.md` 断链 | — | — | — | 引用 EvoPrompt.md | 文件不存在 | 创建或从 sidebar 移除 |

**MAS/Memory/ — 记忆机制：**

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 113B | 空壳 | 标题 + 一句话 | 无概述 | 补充概述 |
| `_sidebar.md` | 1行 | 30B | 空 | 只有 Back to MAS | 无内容条目 | 补充文章后更新 |
| **无文章** | — | — | — | — | 整个子文件夹无实质内容 | 写 1 篇记忆机制入门文章 |

**MAS/FailAndWhy/ — 失败分析：**

| 文件 | 行数 | 大小 | 质量 | 当前内容 | 问题 | 预期改写 |
|------|------|------|------|----------|------|----------|
| `README.md` | 4行 | 125B | 空壳 | 标题 + 一句话 | 无概述 | 补充概述 |
| `_sidebar.md` | 13行 | 588B | ❌全断链 | 列了 12 篇文章 | **12 个文件全部不存在** | 先创建文件再保留，或清空 sidebar |
| **无文章** | — | — | — | — | sidebar 列了 12 篇但一篇都没写 | 至少写 2-3 篇核心文章 |

**MAS/aa/ — 副本目录：**
- 完整复制了 Topology、Prompt、Memory、FailAndWhy 四个子文件夹
- 未被任何导航引用
- **直接删除整个目录**

---

## 五、首页 README.md 状态

当前首页有完整的分类导航，结构清晰。需要在改写完成后：
1. 确认每个链接的描述与实际内容一致
2. 如果有文件夹合并或文章增删，同步更新描述文字

---

## 六、改写优先级建议

| 优先级 | 操作 | 涉及范围 |
|--------|------|----------|
| P0 | 删除冗余（MAS/aa、todo、GFNet.py、重复文件） | 全局 |
| P0 | 修复断链和错误（MAS sidebar、条目名） | MAS、Quantization、MARL |
| P1 | 补充所有 README 概述 | 12 个文件夹 |
| P1 | 填充空文件（BERT、MoE、NLPLearning、safeCritic、DyLAN、AFlow、MIPRO） | LM、DMT、SafeRL、MAS |
| P2 | 优化现有文章可读性 | 全部有内容的文章 |
| P2 | 补充薄弱文件夹（MARL、Transformer_Prediction、MAS/Memory、MAS/FailAndWhy） | 4 个文件夹 |
| P3 | 统一 sidebar 索引、更新首页 | 全局 |



