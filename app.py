import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DB_PATH = os.getenv("KB_DB_PATH", "kb.sqlite3")
MODEL_NAME = os.getenv("KB_MODEL", "gpt-5.4-mini")
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("KB_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "120"))


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clamp_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def safe_json_loads(text: str, fallback: Optional[dict] = None) -> dict:
    fallback = fallback or {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return fallback


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\r\n", "\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paras:
        if len(para) > chunk_size:
            # split long paragraphs by sentences
            sentences = re.split(r"(?<=[.!?。！？])\s+", para)
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(buf) + len(sent) + 1 <= chunk_size:
                    buf = (buf + " " + sent).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sent.strip()
            continue

        if len(buf) + len(para) + 2 <= chunk_size:
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para

    if buf:
        chunks.append(buf)

    # light overlap to preserve context
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    final_chunks: List[str] = []
    prev_tail = ""
    for ch in chunks:
        merged = (prev_tail + "\n\n" + ch).strip() if prev_tail else ch
        final_chunks.append(merged)
        prev_tail = ch[-overlap:] if len(ch) > overlap else ch
    return final_chunks


class KBStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def conn(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self):
        with self.conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            con.commit()

    def insert_document(
        self,
        source: str,
        title: str,
        raw_content: str,
        summary: str,
        tags: List[str],
    ) -> int:
        ts = now_iso()
        with self.conn() as con:
            cur = con.execute(
                """
                INSERT INTO documents (source, title, raw_content, summary, tags_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source, title, raw_content, summary, json.dumps(tags, ensure_ascii=False), ts, ts),
            )
            con.commit()
            return int(cur.lastrowid)

    def insert_chunks(self, document_id: int, chunks: List[str], embeddings: List[List[float]]):
        ts = now_iso()
        with self.conn() as con:
            for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
                con.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, idx, content, json.dumps(emb), ts),
                )
            con.commit()

    def insert_tasks(self, document_id: int, tasks: List[str]):
        ts = now_iso()
        with self.conn() as con:
            for title in tasks:
                if title.strip():
                    con.execute(
                        """
                        INSERT INTO tasks (document_id, title, status, created_at)
                        VALUES (?, ?, 'open', ?)
                        """,
                        (document_id, title.strip(), ts),
                    )
            con.commit()

    def list_documents(self, limit: int = 50):
        with self.conn() as con:
            rows = con.execute(
                """
                SELECT id, source, title, summary, tags_json, created_at
                FROM documents
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, document_id: int):
        with self.conn() as con:
            doc = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not doc:
                return None
            chunks = con.execute(
                """
                SELECT chunk_index, content, embedding_json
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
            tasks = con.execute(
                """
                SELECT id, title, status, created_at
                FROM tasks
                WHERE document_id = ?
                ORDER BY id ASC
                """,
                (document_id,),
            ).fetchall()
            result = dict(doc)
            result["tags"] = json.loads(result.pop("tags_json"))
            result["chunks"] = [dict(r) for r in chunks]
            result["tasks"] = [dict(r) for r in tasks]
            return result

    def search_chunks(self, query_embedding: List[float], limit: int = TOP_K):
        with self.conn() as con:
            rows = con.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding_json, d.title, d.source
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                """
            ).fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, emb)
            scored.append((score, dict(row)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]


class BaseLLM:
    def summarize(self, source: str, title: str, content: str) -> Dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.client = OpenAI()
        self.model = MODEL_NAME
        self.embed_model = EMBED_MODEL

    def summarize(self, source: str, title: str, content: str) -> Dict[str, Any]:
        prompt = f"""
你是一个个人知识库整理 agent。
请从输入内容中提取一个适合存入知识库的结构化结果，严格输出 JSON，不要输出多余文本。

字段要求：
- title: 简洁标题，若原标题已经合适可直接沿用
- summary: 3~5 句中文摘要，尽量保留事实，不要空泛
- tags: 3~6 个中文标签数组
- tasks: 0~5 条可执行待办数组，若没有则返回空数组
- notes: 1~3 条补充说明数组，若没有则返回空数组

输入来源：{source}
原始标题：{title}
内容：
{content}
"""
        response = self.client.responses.create(model=self.model, input=prompt)
        data = safe_json_loads(response.output_text, {})
        return {
            "title": data.get("title") or title or "Untitled",
            "summary": data.get("summary") or clamp_text(content, 220),
            "tags": data.get("tags") or [],
            "tasks": data.get("tasks") or [],
            "notes": data.get("notes") or [],
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in response.data]

    def answer(self, question: str, context: str) -> str:
        prompt = f"""
你是一个个人知识库问答 agent。
只根据给定上下文回答；如果上下文不足以确定答案，明确说明“不确定”，并指出缺少什么信息。
回答要简洁、直接、中文输出。

问题：
{question}

上下文：
{context}
"""
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()


class OfflineLLM(BaseLLM):
    """Fallback for demo/testing without API key."""

    def summarize(self, source: str, title: str, content: str) -> Dict[str, Any]:
        words = re.findall(r"[\w\u4e00-\u9fff]+", content)
        tags = []
        if "论文" in content or "paper" in content.lower():
            tags.append("论文")
        if "todo" in content.lower() or "待办" in content or "任务" in content:
            tags.append("待办")
        if "代码" in content or "python" in content.lower():
            tags.append("编程")
        summary = clamp_text(content, 220)
        return {
            "title": title or (words[0] if words else "Untitled"),
            "summary": summary,
            "tags": tags[:4] or ["未分类"],
            "tasks": [],
            "notes": [f"来源：{source}"],
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Deterministic tiny bag-of-characters embedding for offline demo
        vectors = []
        for t in texts:
            vec = [0.0] * 16
            for ch in t.lower():
                vec[ord(ch) % 16] += 1.0
            vectors.append(vec)
        return vectors

    def answer(self, question: str, context: str) -> str:
        return f"离线模式下的演示回答。问题：{question}\n参考上下文：{clamp_text(context, 400)}"


def build_llm() -> BaseLLM:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    return OfflineLLM()


@dataclass
class IngestResult:
    document_id: int
    title: str
    summary: str
    tags: List[str]
    tasks: List[str]


class KnowledgeCoordinator:
    def __init__(self, store: KBStore, llm: BaseLLM):
        self.store = store
        self.llm = llm

    def ingest(self, source: str, title: str, content: str) -> IngestResult:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("content is empty")

        analysis = self.llm.summarize(source=source, title=title, content=cleaned)
        final_title = (analysis.get("title") or title or "Untitled").strip()
        summary = (analysis.get("summary") or clamp_text(cleaned)).strip()
        tags = [str(x).strip() for x in (analysis.get("tags") or []) if str(x).strip()]
        tasks = [str(x).strip() for x in (analysis.get("tasks") or []) if str(x).strip()]

        doc_id = self.store.insert_document(
            source=source,
            title=final_title,
            raw_content=cleaned,
            summary=summary,
            tags=tags,
        )

        chunks = chunk_text(cleaned)
        embeddings = self.llm.embed(chunks) if chunks else []
        if chunks:
            self.store.insert_chunks(doc_id, chunks, embeddings)
        if tasks:
            self.store.insert_tasks(doc_id, tasks)

        return IngestResult(
            document_id=doc_id,
            title=final_title,
            summary=summary,
            tags=tags,
            tasks=tasks,
        )

    def retrieve_context(self, question: str) -> str:
        q_emb = self.llm.embed([question])[0]
        hits = self.store.search_chunks(q_emb, limit=TOP_K)
        blocks = []
        for rank, (score, row) in enumerate(hits, start=1):
            blocks.append(
                f"[{rank}] score={score:.3f} doc#{row['document_id']} {row['title']} ({row['source']})\n"
                f"{row['content']}"
            )
        return "\n\n".join(blocks) if blocks else "（无可用上下文）"

    def answer(self, question: str) -> Dict[str, Any]:
        context = self.retrieve_context(question)
        answer = self.llm.answer(question, context)
        return {"question": question, "answer": answer, "context": context}


store = KBStore()
llm = build_llm()
coordinator = KnowledgeCoordinator(store, llm)
app = FastAPI(title="Personal Knowledge Base MVP", version="0.1.0")


class IngestRequest(BaseModel):
    source: str = Field(default="manual", examples=["web", "wechat", "email", "manual"])
    title: str = Field(default="")
    content: str


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "time": now_iso(), "model": MODEL_NAME if isinstance(llm, OpenAILLM) else "offline"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        result = coordinator.ingest(req.source, req.title, req.content)
        return {
            "document_id": result.document_id,
            "title": result.title,
            "summary": result.summary,
            "tags": result.tags,
            "tasks": result.tasks,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return coordinator.answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents")
def list_documents(limit: int = 50):
    return {"items": store.list_documents(limit=limit)}


@app.get("/documents/{document_id}")
def get_document(document_id: int):
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@app.get("/")
def root():
    return {
        "name": "Personal Knowledge Base MVP",
        "endpoints": ["/health", "/ingest", "/ask", "/documents", "/documents/{id}"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
