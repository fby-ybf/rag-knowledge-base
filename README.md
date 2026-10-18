# rag-knowledge-base

企业知识库问答（RAG）的最小可运行复现：文档上传 → 分块 → 向量化 → 检索 → 生成答案，
附 Docker 部署与 CI。定位为「MaxKB4j 一类企业 RAG 知识库」的练习实现。

技术栈：Python / FastAPI / scikit-learn(TF-IDF 向量) / OpenAI 兼容接口（可切换离线模式）

## 功能

- 文档上传（.txt / .md，按段落 + 滑动窗口分块，带重叠）
- 向量化入库（默认 TF-IDF，无需下载模型、可离线运行；预留 sentence-transformers 接口）
- 相似度检索 + 元数据过滤（按文档名 / 标签）
- 问答：调用 OpenAI 兼容接口（`OPENAI_BASE_URL` + `OPENAI_API_KEY` + `OPENAI_MODEL`），
  **未配置时自动进入离线模式**：直接基于检索到的上下文拼接带引用的答案，方便无 Key 演示与测试
- 索引持久化（pickle），重启不丢
- `/healthz`、`/stats` 运维端点

## 快速开始

```bash
pip install -r requirements.txt
uvicorn app.api:app --port 8000
# 打开 http://localhost:8000/ 上传文档并提问
```

接入真实大模型：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 或任何兼容端点
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
uvicorn app.api:app --port 8000
```

Docker：

```bash
docker compose up -d --build     # 服务在 8000 端口
```

## 目录

```
app/
  chunker.py    分块（段落优先 + 定长滑窗 + 重叠）
  embedder.py   TF-IDF 向量化（接口可替换为语义向量）
  store.py      内存向量库：相似度检索、元数据过滤、持久化
  llm.py        OpenAI 兼容客户端 + 离线回退
  pipeline.py   索引与问答编排
  api.py        FastAPI 路由与静态页面
  web/index.html 最小前端
tests/          分块 / 检索排序 / 离线问答 / API 测试
```

## 测试

```bash
pip install -r requirements.txt pytest httpx
python -m pytest -q
```

## 设计取舍

- **向量化用 TF-IDF 而非语义向量模型**：保证零外部依赖、CI 与离线环境可复现；
  生产环境把 `embedder.py` 换成 bge / text-embedding 即可，检索接口不变。
- **离线模式**：没有 LLM Key 时也能完整体验「检索 → 引用上下文 → 回答」的链路，
  答案是模板拼接而非模型生成，README 与接口响应里都会标注 `mode: offline`。
