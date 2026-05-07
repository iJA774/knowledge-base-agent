# Personal Knowledge Base MVP

一个“自动化个人知识库”的最小可用版本。

## 结构

- `POST /ingest`：输入一段原始内容，系统会自动摘要、打标签、分块、向量化，并写入 SQLite
- `POST /ask`：基于检索到的知识块进行问答
- `GET /documents`：查看已入库内容
- `GET /documents/{id}`：查看单条内容详情

## 运行

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="你的key"
python app.py
```

然后访问 `http://127.0.0.1:8000/docs`

## 说明

- 代码默认使用 OpenAI 的 Responses API 与 embeddings
- 没有 `OPENAI_API_KEY` 时会自动切换到离线演示模式，方便先跑通流程
- 数据库存到 `kb.sqlite3`

## 示例

```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{"source":"manual","title":"读书笔记","content":"今天记录了RAG系统的几个要点……"}'
```

```bash
curl -X POST http://127.0.0.1:8000/ask   -H "Content-Type: application/json"   -d '{"question":"我之前记录的RAG要点是什么？"}'
```
