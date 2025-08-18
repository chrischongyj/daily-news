import os
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from readability import Document
from tqdm import tqdm

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ---------------------------
# ENV & CONFIG
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️  Set OPENAI_API_KEY in your environment for best results.")

DB_PATH = os.getenv("NEWS_DB", "news_feed.sqlite")
MARKDOWN_OUT = os.getenv("NEWS_MD", "./docs/index.md")
MAX_PER_SOURCE = int(os.getenv("MAX_PER_SOURCE", "20"))  # pull up to N items per source run
MAX_ITEMS_PER_RUN = int(os.getenv("MAX_ITEMS_PER_RUN", "80"))  # global cap per run

# News sources by category (RSS preferred for reliability)
SOURCES = {
    "finance": [
        "https://www.reuters.com/finance/rss",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ Markets (some items may be gated)
        "https://www.ft.com/rss/home",  # FT (summaries often available)
        "https://www.bloomberg.com/feeds/podcasts/etf-report.xml",  # Example; many BBG feeds are limited
    ],
    "startup": [
        "https://techcrunch.com/startups/feed/",
        "https://feeds.feedburner.com/avc",  # Fred Wilson
        "https://news.crunchbase.com/feed/",
    ],
    "tech": [
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://hnrss.org/frontpage",
    ],
    "global": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://rss.cnn.com/rss/edition_world.rss",
        "https://www.reuters.com/world/rss",
    ],
}

# ---------------------------
# DB: simple SQLite store
# ---------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id TEXT PRIMARY KEY,
        url TEXT,
        source TEXT,
        category TEXT,
        title TEXT,
        published_at TEXT,
        author TEXT,
        raw_summary TEXT,
        content TEXT,
        llm_summary TEXT,
        importance INTEGER,
        entities TEXT,
        tickers TEXT,
        created_at TEXT
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_cat_time ON items(category, published_at)")
    conn.commit()
    conn.close()

def upsert_item(item: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cols = ",".join(item.keys())
    placeholders = ",".join(["?"] * len(item))
    # SQLite UPSERT by primary key
    updates = ",".join([f"{k}=excluded.{k}" for k in item.keys() if k != "id"])
    sql = f"INSERT INTO items ({cols}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}"
    c.execute(sql, list(item.values()))
    conn.commit()
    conn.close()

# ---------------------------
# Fetch & extract
# ---------------------------

def hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text_html(s: str) -> str:
    if not s:
        return ""
    soup = BeautifulSoup(s, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def fetch_rss_items() -> List[Dict[str, Any]]:
    all_items = []
    for category, feeds in SOURCES.items():
        for feed in feeds:
            try:
                parsed = feedparser.parse(feed)
                entries = parsed.entries[:MAX_PER_SOURCE]
                for e in entries:
                    link = getattr(e, "link", "")
                    title = clean_text_html(getattr(e, "title", ""))
                    summary = clean_text_html(getattr(e, "summary", ""))
                    author = clean_text_html(getattr(e, "author", "")) if hasattr(e, "author") else ""
                    published = None
                    if hasattr(e, "published"):
                        try:
                            published = dateparser.parse(e.published).astimezone(timezone.utc).isoformat()
                        except Exception:
                            published = None
                    # fallback to updated
                    if not published and hasattr(e, "updated"):
                        try:
                            published = dateparser.parse(e.updated).astimezone(timezone.utc).isoformat()
                        except Exception:
                            published = None
                    # ID based on link or title+source
                    pid = hash_id(link or f"{title}|{feed}")
                    all_items.append({
                        "id": pid,
                        "url": link,
                        "source": feed,
                        "category": category,
                        "title": title,
                        "published_at": published or datetime.now(timezone.utc).isoformat(),
                        "author": author,
                        "raw_summary": summary,
                    })
            except Exception as ex:
                print(f"RSS error for {feed}: {ex}")
    # global cap
    return all_items[:MAX_ITEMS_PER_RUN]

def fetch_and_read(url: str, timeout=12) -> str:
    """Fetch full article and extract main text with Readability."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (NewsAnalyst/1.0)"})
        if r.status_code != 200:
            return ""
        doc = Document(r.text)
        html = doc.summary()
        txt = clean_text_html(html)
        # if empty, fall back to whole page text
        if not txt:
            txt = clean_text_html(r.text)
        return txt.strip()
    except Exception:
        return ""

# ---------------------------
# LLM helpers (LangChain)
# ---------------------------

def get_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    # You can set model via env: MODEL_NAME
    model_name = os.getenv("MODEL_NAME", model)
    return ChatOpenAI(model=model_name, temperature=0.2)

def llm_summarize_rank_entities(llm: ChatOpenAI, text: str, category: str) -> Dict[str, Any]:
    sys = SystemMessage(content=(
        "You are an analyst who writes tight, factual, finance- and tech-savvy briefs. "
        "Output JSON with keys: summary (<=60 words), importance (1-5), entities (comma list), tickers (comma list). "
        "importance=impact on investors/founders/tech workers. If unsure about tickers, leave empty."
    ))
    user = HumanMessage(content=f"Category: {category}\n\nArticle:\n{text[:8000]}")
    resp = llm.invoke([sys, user]).content.strip()
    # Very lightweight JSON-ish extraction (robust to chatty LLMs)
    # Expect lines like: summary: ..., importance: 4, entities: A,B,C, tickers: MSFT,TSLA
    out = {"summary": "", "importance": 3, "entities": "", "tickers": ""}
    try:
        # crude parse
        lower = resp.lower()
        # find keys via simple heuristics
        def after(label):
            idx = lower.find(label)
            if idx == -1:
                return ""
            return resp[idx+len(label):].splitlines()[0].strip(" -:;")
        if "summary" in lower:
            out["summary"] = after("summary").strip()
        if "importance" in lower:
            imp = after("importance").split()[0].strip(",.")
            out["importance"] = max(1, min(5, int("".join([c for c in imp if c.isdigit()]) or "3")))
        if "entities" in lower:
            out["entities"] = after("entities").strip().strip(",")
        if "tickers" in lower:
            out["tickers"] = after("tickers").strip().strip(",")
    except Exception:
        pass
    # Fallback if LLM returned raw JSON already
    if resp.startswith("{") and "summary" in resp:
        try:
            import json
            maybe = json.loads(resp)
            out.update({k: maybe.get(k, out[k]) for k in out.keys()})
        except Exception:
            pass
    return out

# ---------------------------
# LangGraph state & nodes
# ---------------------------

class Item(TypedDict, total=False):
    id: str
    url: str
    source: str
    category: str
    title: str
    published_at: str
    author: str
    raw_summary: str
    content: str
    llm_summary: str
    importance: int
    entities: str
    tickers: str

class PipelineState(TypedDict):
    items: List[Item]
    processed: List[Item]

def node_fetch(state: PipelineState) -> PipelineState:
    fetched = fetch_rss_items()
    return {"items": fetched, "processed": []}

def node_dedupe_and_content(state: PipelineState) -> PipelineState:
    seen = set()
    unique = []
    for it in state["items"]:
        key = it["id"]
        if key in seen:
            continue
        seen.add(key)
        # fetch full text (best-effort)
        content = fetch_and_read(it["url"]) if it.get("url") else ""
        it["content"] = content or it.get("raw_summary", "")
        unique.append(it)
    return {"items": unique, "processed": state["processed"]}

def node_analyze(state: PipelineState) -> PipelineState:
    llm = get_llm()
    out: List[Item] = []
    for it in tqdm(state["items"], desc="Analyzing"):
        text = f"{it['title']}\n\n{it.get('content','')}"
        res = llm_summarize_rank_entities(llm, text, it["category"])
        it["llm_summary"] = res.get("summary", "")
        it["importance"] = res.get("importance", 3)
        it["entities"] = res.get("entities", "")
        it["tickers"] = res.get("tickers", "")
        out.append(it)
    return {"items": out, "processed": state["processed"]}

def node_publish(state: PipelineState) -> PipelineState:
    now = datetime.now(timezone.utc).isoformat()
    for it in state["items"]:
        record = {
            "id": it["id"],
            "url": it.get("url", ""),
            "source": it.get("source", ""),
            "category": it.get("category", ""),
            "title": it.get("title", ""),
            "published_at": it.get("published_at", ""),
            "author": it.get("author", ""),
            "raw_summary": it.get("raw_summary", ""),
            "content": it.get("content", ""),
            "llm_summary": it.get("llm_summary", ""),
            "importance": int(it.get("importance", 3)),
            "entities": it.get("entities", ""),
            "tickers": it.get("tickers", ""),
            "created_at": now,
        }
        upsert_item(record)
    # Also write a Markdown digest
    write_markdown_digest(limit=50)
    return {"items": [], "processed": state["items"]}

def write_markdown_digest(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT title, category, llm_summary, importance, tickers, entities, url, published_at
        FROM items
        ORDER BY importance DESC, published_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    lines = ["# AI Curated News Analyst — Latest\n"]
    for title, category, summary, importance, tickers, entities, url, published_at in rows:
        when = published_at.split("T")[0] if published_at else ""
        tick = f" — Tickers: {tickers}" if tickers else ""
        ent  = f" — Entities: {entities}" if entities else ""
        lines.append(f"**[{title}]({url})**  \n*{category}* · {when} · **Imp {importance}/5**{tick}{ent}\n\n{summary}\n")
    with open(MARKDOWN_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📝 Wrote digest to {MARKDOWN_OUT}")

# ---------------------------
# Build the graph
# ---------------------------

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("fetch", node_fetch)
    graph.add_node("dedupe_content", node_dedupe_and_content)
    graph.add_node("analyze", node_analyze)
    graph.add_node("publish", node_publish)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "dedupe_content")
    graph.add_edge("dedupe_content", "analyze")
    graph.add_edge("analyze", "publish")
    graph.add_edge("publish", END)
    return graph.compile()

# ---------------------------
# Optional: FastAPI read-only feed
# ---------------------------

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Curated News Analyst")

@app.get("/feed")
def get_feed(category: Optional[str] = None, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if category:
        c.execute("""
            SELECT title, category, llm_summary, importance, tickers, entities, url, published_at
            FROM items
            WHERE category=?
            ORDER BY importance DESC, published_at DESC
            LIMIT ?
        """, (category, limit))
    else:
        c.execute("""
            SELECT title, category, llm_summary, importance, tickers, entities, url, published_at
            FROM items
            ORDER BY importance DESC, published_at DESC
            LIMIT ?
        """, (limit,))
    rows = c.fetchall()
    conn.close()
    items = []
    for title, category, summary, importance, tickers, entities, url, published_at in rows:
        items.append({
            "title": title,
            "category": category,
            "summary": summary,
            "importance": importance,
            "tickers": tickers,
            "entities": entities,
            "url": url,
            "published_at": published_at
        })
    return JSONResponse(items)

# ---------------------------
# CLI
# ---------------------------

def run_once():
    init_db()
    graph = build_graph()
    state: PipelineState = {"items": [], "processed": []}
    graph.invoke(state)
    print("✅ Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Curated News Analyst")
    parser.add_argument("--run", action="store_true", help="Run the pipeline once")
    parser.add_argument("--serve", action="store_true", help="Serve FastAPI on :8000")
    args = parser.parse_args()
    if args.run:
        run_once()
    if args.serve:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
