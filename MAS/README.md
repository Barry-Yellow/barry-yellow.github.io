# 多智能体系统 (Multi-Agent System)

本部分记录了基于大语言模型（LLM）的多智能体系统（MAS）方向的研究笔记。与传统 MARL 不同，LLM-based MAS 关注的是如何利用大语言模型构建能够协作完成复杂任务的智能体团队。

## 研究子方向

- **拓扑优化**：研究智能体之间的连接结构和通信拓扑如何影响系统性能。包括 DyLAN（动态 LLM-Agent 网络）、AFlow（基于 MCTS 的自动化工作流搜索）等工作。
- **提示词优化**：研究如何自动优化多智能体系统中各智能体的提示词。包括 MIPRO（贝叶斯优化多阶段指令）、MAS-GPT（训练 LLM 构建多智能体系统）等工作。
- **失败分析与修复**：研究多智能体系统为何失败、如何定位失败原因、以及如何从失败中恢复。这是 MAS 可靠性研究的核心方向。
- **记忆机制**：研究如何为多智能体系统设计有效的记忆模块，使智能体能够积累和利用历史经验。

## 领域经典论文与项目

| 论文/项目 | 年份 | 核心贡献 |
|-----------|------|----------|
| [AutoGen](https://github.com/microsoft/autogen) (Microsoft) | 2023 | 开源多智能体对话框架，支持灵活的智能体编排和人机协作 |
| [MetaGPT](https://github.com/geekan/MetaGPT) | 2023 | 将 SOP（标准操作流程）编码到多智能体协作中，模拟软件公司工作流 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 2024 | 基于角色的多智能体编排框架，强调任务分工和流程控制 |
| [ChatDev](https://github.com/OpenBMB/ChatDev) | 2023 | 用多智能体模拟软件开发流程（需求→设计→编码→测试） |
| Camel (Li et al.) | 2023 | 提出角色扮演（Role-Playing）框架，研究两个 LLM 智能体的自主协作 |
| AgentVerse (Chen et al.) | 2023 | 动态智能体组合框架，支持专家招募和任务分解 |
| DyLAN (Liu et al.) | 2024 | 基于贡献度排名的动态智能体网络，自动调整通信拓扑 |
| AFlow (Zhang et al.) | 2024 | 用 MCTS 搜索最优工作流，以代码形式表示智能体协作流程 |

[](_sidebar.md ':include')
