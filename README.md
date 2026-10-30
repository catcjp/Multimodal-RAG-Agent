# 🧠 多模态智能文档分析助手 (Multimodal RAG + Agent)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![LLM](https://img.shields.io/badge/LLM-Qwen--Max-green)
![Multimodal](https://img.shields.io/badge/Model-Qwen--VL-orange)

基于大模型（DashScope / Qwen）构建的智能文档问答系统。本项目不仅实现了基础的 RAG，还针对**复杂文档解析、Agent 工具路由、专有名词精准召回**以及**大模型安全防御**进行了深度的工程化改造。

👉 **[点击这里体验在线 Demo](https://multimodal-rag-agent.streamlit.app/)**

---

## ✨ 核心特性 (Features)

- 🖼️ **多模态数据摄入**：突破传统纯文本提取限制。集成 `pdfplumber` 提取复杂表格，`PyMuPDF` 抽取图像，并调用 `qwen-vl` 视觉大模型进行图像文字识别（OCR），实现非结构化数据的语义融合。
- 🤖 **Agent 智能决策路由**：摒弃脆弱的正则/字符串匹配。基于强约束 System Prompt 与 JSON Schema，让大模型自主决定调用 `search_documents`（本地检索）或 `write_file`（生成文件）工具，实现从“被动回答”到“主动执行”的跨越。
- 🔍 **混合检索兜底 (Hybrid Retrieval)**：针对长文档中“专有名词、作者、邮箱”等细节信息的语义漂移问题，手写**基于规则的强制召回策略**。在向量检索失效时，强制将首页元数据加入上下文，大幅提升关键信息的召回率。
- 🛡️ **大模型安全防御 (Security)**：针对 Prompt Injection（提示词注入）攻击进行了防御加固。即使面对“忽略指令”的诱导，系统依然能严守边界，拒绝执行与文档无关的代码编写或创作任务。
- ⚡ **工程级高可用架构**：
  - **持久化缓存**：基于 JSON + Pickle 实现多文档隔离缓存，冷启动响应时间从数十秒压缩至**秒级**。
  - **优雅降级**：全局 `try-except` 与 API 状态码拦截，防止外部接口异常导致 Web 前端崩溃。
  - **无状态通信管理**：利用 Streamlit `session_state` 解决 Web 刷新导致的数据丢失痛点。

---

## 🏗️ 系统架构图 (Architecture)

```mermaid
graph TD
    User((用户)) -->|上传 PDF & 提问| Web[Streamlit Web 界面]
    Web --> |session_state 状态管理| Agent[Agent 路由中枢]
    
    subgraph Agent Core
        Agent --> |Prompt + JSON Schema 约束| Qwen(Qwen-Max 决策)
        Qwen --> |返回 JSON 指令| Agent
        Agent -->|action: search_documents| Vector[向量检索 / 规则兜底]
        Agent -->|action: write_file| FileIO[生成 txt 文件]
    end
    
    Web --> |触发解析| Extractor[多模态解析管线]
    subgraph 数据摄入
        Extractor --> Txt[Pypdf 提取文本]
        Extractor --> Table[pdfplumber 提取表格]
        Extractor --> Img[Qwen-VL 识别图像]
    end
    
    Extractor --> |生成缓存与向量| Cache[(本地 JSON + Pickle 缓存)]
    Vector --> |结合上下文| FinalLLM(Qwen-Max 生成)
    FinalLLM --> |返回高质量回答| Web
