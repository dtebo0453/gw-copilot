from __future__ import annotations

import hashlib
import os
import sqlite3
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Lightweight documentation retrieval using SQLite FTS5.
# This avoids external dependencies and provides predictable, local "RAG" behavior.

SUPPORTED_EXTS = {".md", ".txt", ".rst", ".html", ".htm"}


@dataclass
class DocHit:
    source: str           # relative path
    title: str
    snippet: str
    score: float


def _safe_rel(ws_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ws_root.resolve()))
    except Exception:
        return str(p)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _iter_doc_files(docs_root: Path) -> Iterable[Path]:
    for p in docs_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _read_text(p: Path, max_chars: int = 2_000_000) -> str:
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return t[:max_chars]


def _chunk_text(text: str, *, chunk_chars: int = 3500, overlap: int = 300) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
        doc_path,
        title,
        chunk_id UNINDEXED,
        content
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs_meta (
        docs_root TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL
    );
    """)
    conn.commit()


def _fingerprint_docs(ws_root: Path, docs_root: Path) -> str:
    # fingerprint includes file relpaths + mtimes + sizes
    parts: List[str] = []
    for p in sorted(_iter_doc_files(docs_root), key=lambda x: str(x)):
        try:
            st = p.stat()
            parts.append(f"{_safe_rel(ws_root,p)}|{st.st_size}|{int(st.st_mtime)}")
        except Exception:
            parts.append(f"{_safe_rel(ws_root,p)}|ERR")
    return _hash_text("\n".join(parts))


def build_or_refresh_index(ws_root: Path, docs_root: Path) -> Optional[Path]:
    """Build/refresh a docs index under ws_root/.gw_copilot/docs_index.sqlite.

    Returns the sqlite path if indexing was performed or already available.
    Returns None if docs_root doesn't exist / has no supported docs.
    """
    if not docs_root.exists():
        return None

    files = list(_iter_doc_files(docs_root))
    if not files:
        return None

    index_dir = ws_root / ".gw_copilot"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "docs_index.sqlite"

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        fp = _fingerprint_docs(ws_root, docs_root)

        row = conn.execute("SELECT fingerprint FROM docs_meta WHERE docs_root=?", (str(docs_root.resolve()),)).fetchone()
        if row and row[0] == fp:
            return db_path

        # Rebuild
        conn.execute("DELETE FROM docs_fts;")
        conn.execute("DELETE FROM docs_meta WHERE docs_root=?", (str(docs_root.resolve()),))

        for p in files:
            rel = _safe_rel(ws_root, p)
            text = _read_text(p)
            # very simple title heuristic: first non-empty line (trimmed) or filename
            title = ""
            for ln in text.splitlines()[:20]:
                ln = ln.strip()
                if ln:
                    title = ln.lstrip("# ").strip()
                    break
            if not title:
                title = p.name

            chunks = _chunk_text(text)
            for ci, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO docs_fts(doc_path,title,chunk_id,content) VALUES (?,?,?,?)",
                    (rel, title, str(ci), chunk),
                )

        conn.execute(
            "INSERT INTO docs_meta(docs_root,fingerprint) VALUES (?,?)",
            (str(docs_root.resolve()), fp),
        )
        conn.commit()
        return db_path
    finally:
        conn.close()


def search_docs(ws_root: Path, docs_root: Path, query: str, *, top_k: int = 5) -> List[DocHit]:
    db_path = build_or_refresh_index(ws_root, docs_root)
    if not db_path:
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        # FTS5 bm25() is available via `bm25(docs_fts)` in many builds; if not, fall back to rank.
        # We'll compute a score as negative bm25 (lower is better), then invert.
        sql = """
        SELECT doc_path, title,
               snippet(docs_fts, 3, '[', ']', '…', 12) AS snip,
               bm25(docs_fts) AS score
        FROM docs_fts
        WHERE docs_fts MATCH ?
        ORDER BY score ASC
        LIMIT ?;
        """
        try:
            rows = conn.execute(sql, (query, int(top_k))).fetchall()
        except sqlite3.OperationalError:
            # No bm25() in this build
            sql2 = """
            SELECT doc_path, title,
                   snippet(docs_fts, 3, '[', ']', '…', 12) AS snip,
                   0.0 AS score
            FROM docs_fts
            WHERE docs_fts MATCH ?
            LIMIT ?;
            """
            rows = conn.execute(sql2, (query, int(top_k))).fetchall()

        hits: List[DocHit] = []
        for doc_path, title, snip, score in rows:
            # Convert lower-is-better score to higher-is-better
            try:
                s = float(score)
                score_adj = 1.0 / (1.0 + max(0.0, s))
            except Exception:
                score_adj = 0.0
            hits.append(DocHit(source=str(doc_path), title=str(title), snippet=str(snip), score=score_adj))
        return hits
    finally:
        conn.close()


def search_workspace_docs(ws_root: Path, query: str, *, k: int = 5) -> List[dict]:
    """Search docs in conventional workspace locations.

    The GW Copilot uses local retrieval for documentation to keep answers grounded
    without requiring internet access.

    Docs roots searched (in order):
      1) <workspace>/docs
      2) <workspace>/documentation
      3) built-in quick reference shipped with the app (small fallback)

    Returns a JSON-serializable list of hits.
    """
    roots: List[Path] = []
    for name in ("docs", "documentation"):
        p = ws_root / name
        if p.exists():
            roots.append(p)

    # Built-in fallback (small, conservative). This is not a replacement for official docs.
    try:
        builtin = Path(__file__).resolve().parent / "builtin_docs"
        if builtin.exists():
            roots.append(builtin)
    except Exception:
        pass

    hits: List[DocHit] = []
    for r in roots:
        try:
            hits.extend(search_docs(ws_root, r, query, top_k=max(2, k)))
        except Exception:
            continue

    # de-dup by (source,title,snippet)
    seen = set()
    out: List[dict] = []
    for h in sorted(hits, key=lambda x: (-x.score, x.source)):
        key = (h.source, h.title, h.snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": h.source,
            "title": h.title,
            "snippet": h.snippet,
            "score": h.score,
        })
        if len(out) >= k:
            break
    return out


# -------------------------
# Controlled web retrieval
# -------------------------

# IMPORTANT: This is a *backend-controlled* web retriever (not the LLM directly).
# It is designed for:
#   - official docs domains only (allowlist)
#   - tight timeouts, redirect limits, size limits
#   - caching to disk for performance + reproducibility
#
# Enable with: GW_ENABLE_WEB_DOCS=1


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in {"script", "noscript"}:
            self._in_script = True
        if t == "style":
            self._in_style = True

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in {"script", "noscript"}:
            self._in_script = False
        if t == "style":
            self._in_style = False

    def handle_data(self, data: str):
        if self._in_script or self._in_style:
            return
        if data:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(self._parts)


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "") or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _allowlist_hosts() -> List[str]:
    raw = str(os.environ.get(
        "GW_DOCS_ALLOWLIST",
        "modflow6.readthedocs.io,water.usgs.gov,usgs.gov,flopy.readthedocs.io,github.com",
    ) or "")
    hosts = [h.strip().lower() for h in raw.split(",") if h.strip()]
    return hosts


def _url_allowed(url: str) -> bool:
    try:
        u = urllib.parse.urlparse(url)
        host = (u.hostname or "").lower()
        if not host:
            return False
        for allowed in _allowlist_hosts():
            if host == allowed or host.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False


def _web_cache_path(ws_root: Path) -> Path:
    d = ws_root / ".gw_copilot"
    d.mkdir(parents=True, exist_ok=True)
    return d / "web_docs_cache.sqlite"


def _web_cache_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_pages (
            url TEXT PRIMARY KEY,
            fetched_ts INTEGER NOT NULL,
            etag TEXT,
            last_modified TEXT,
            title TEXT,
            text TEXT
        );
        """
    )
    conn.commit()


def _web_cache_get(conn: sqlite3.Connection, url: str, *, max_age_days: int) -> Optional[Dict[str, str]]:
    row = conn.execute(
        "SELECT fetched_ts, etag, last_modified, title, text FROM web_pages WHERE url=?",
        (url,),
    ).fetchone()
    if not row:
        return None
    fetched_ts, etag, last_modified, title, text = row
    try:
        age = int(time.time()) - int(fetched_ts)
    except Exception:
        age = 10**12
    if age > max_age_days * 86400:
        return None
    return {
        "etag": etag or "",
        "last_modified": last_modified or "",
        "title": title or "",
        "text": text or "",
    }


def _web_cache_put(conn: sqlite3.Connection, url: str, *, etag: str, last_modified: str, title: str, text: str) -> None:
    conn.execute(
        """
        INSERT INTO web_pages(url,fetched_ts,etag,last_modified,title,text)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
          fetched_ts=excluded.fetched_ts,
          etag=excluded.etag,
          last_modified=excluded.last_modified,
          title=excluded.title,
          text=excluded.text;
        """,
        (url, int(time.time()), etag, last_modified, title, text),
    )
    conn.commit()


def _fetch_url_text(url: str, *, timeout_s: int = 8, max_bytes: int = 1_500_000, max_redirects: int = 3) -> Dict[str, str]:
    """Fetch a URL and return {title,text,etag,last_modified}.

    Uses standard library urllib to avoid extra deps.
    """
    if not _url_allowed(url):
        raise ValueError("URL host not allowed")

    cur = url
    redirects = 0
    etag = ""
    last_mod = ""

    while True:
        req = urllib.request.Request(cur, headers={
            "User-Agent": "GW-Copilot/1.0 (+docs retrieval)",
            "Accept": "text/html,text/plain,*/*",
        })
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            code = getattr(resp, "status", 200)
            final_url = resp.geturl()
            # Handle redirects explicitly for safety
            if code in {301, 302, 303, 307, 308} and redirects < max_redirects:
                loc = resp.headers.get("Location")
                if not loc:
                    break
                nxt = urllib.parse.urljoin(final_url, loc)
                if not _url_allowed(nxt):
                    raise ValueError("redirect target not allowed")
                redirects += 1
                cur = nxt
                continue

            etag = resp.headers.get("ETag", "") or ""
            last_mod = resp.headers.get("Last-Modified", "") or ""

            raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]

            # Decode best-effort
            try:
                text_raw = raw.decode("utf-8", errors="replace")
            except Exception:
                text_raw = raw.decode(errors="replace")

            # Title heuristic
            m = re.search(r"<title[^>]*>(.*?)</title>", text_raw, flags=re.IGNORECASE | re.DOTALL)
            title = (m.group(1).strip() if m else "")

            # Extract visible text
            parser = _HTMLTextExtractor()
            try:
                parser.feed(text_raw)
            except Exception:
                pass
            text = parser.text()
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return {
                "url": final_url,
                "etag": etag,
                "last_modified": last_mod,
                "title": title or final_url,
                "text": text,
            }


_MF6_URLS_BY_EXT = {
    ".chd": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-chd.html"],
    ".wel": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-wel.html"],
    ".ghb": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-ghb.html"],
    ".riv": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-riv.html"],
    ".drn": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-drn.html"],
    ".npf": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-npf.html"],
    ".sto": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-sto.html"],
    ".ic":  ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-ic.html"],
    ".oc":  ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-oc.html"],
    ".dis": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-dis.html"],
    ".disv": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-disv.html"],
    ".disu": ["https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-disu.html"],
    ".tdis": ["https://modflow6.readthedocs.io/en/latest/_mf6io/sim-tdis.html"],
    ".ims": ["https://modflow6.readthedocs.io/en/latest/_mf6io/sim-ims.html"],
}


def _candidate_doc_urls_from_question(question: str, *, explicit_files: Optional[List[str]] = None) -> List[str]:
    q = (question or "").lower()
    urls: List[str] = []

    # If user referenced a known file extension, prefer the official MF6 IO page.
    if explicit_files:
        for p in explicit_files:
            ext = Path(p).suffix.lower()
            urls.extend(_MF6_URLS_BY_EXT.get(ext, []))
            # Special-case name files: simulation name file vs model name file
            if ext == ".nam":
                bn = Path(p).name.lower()
                if bn.startswith("mfsim"):
                    urls.append("https://modflow6.readthedocs.io/en/latest/_mf6io/mfsim-nam.html")
                else:
                    urls.append("https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-nam.html")

    # Heuristic package tokens in question
    token_map = {
        "chd": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-chd.html",
        "wel": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-wel.html",
        "ghb": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-ghb.html",
        "dis": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-dis.html",
        "npf": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-npf.html",
        "sto": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-sto.html",
        "ic": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-ic.html",
        "oc": "https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-oc.html",
        "tdis": "https://modflow6.readthedocs.io/en/latest/_mf6io/sim-tdis.html",
        "ims": "https://modflow6.readthedocs.io/en/latest/_mf6io/sim-ims.html",
        "mfsim": "https://modflow6.readthedocs.io/en/latest/_mf6io/mfsim-nam.html",
        "name file": "https://modflow6.readthedocs.io/en/latest/_mf6io/mfsim-nam.html",
    }
    for tok, u in token_map.items():
        if tok in q:
            urls.append(u)

    # USGS landing page (broad)
    if any(w in q for w in ["modflow 6", "mf6", "documentation", "manual"]):
        urls.append("https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model")

    # De-dup preserving order
    out: List[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out[:8]


def _chunk_for_snippets(text: str, *, chunk_chars: int = 2000, overlap: int = 150) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


def _score_chunk(chunk: str, query: str) -> float:
    # Very small, predictable scoring: count of query tokens occurrences.
    q = re.sub(r"[^a-z0-9_ ]+", " ", (query or "").lower())
    toks = [t for t in q.split() if len(t) >= 3]
    if not toks:
        return 0.0
    c = chunk.lower()
    score = 0.0
    for t in toks:
        score += float(c.count(t))
    return score


def search_web_docs(
    ws_root: Path,
    query: str,
    *,
    k: int = 5,
    explicit_files: Optional[List[str]] = None,
) -> List[dict]:
    """Controlled web retrieval for official docs.

    Returns JSON-serializable hits with fields: source,title,snippet,score.
    """
    if not _env_bool("GW_ENABLE_WEB_DOCS", default=False):
        return []

    urls = _candidate_doc_urls_from_question(query, explicit_files=explicit_files)
    if not urls:
        return []

    try:
        max_age_days = int(os.environ.get("GW_WEB_DOCS_CACHE_DAYS", "30") or "30")
    except Exception:
        max_age_days = 30
    try:
        timeout_s = int(os.environ.get("GW_WEB_DOCS_TIMEOUT", "8") or "8")
    except Exception:
        timeout_s = 8
    try:
        max_pages = int(os.environ.get("GW_WEB_DOCS_MAX_PAGES", "3") or "3")
    except Exception:
        max_pages = 3

    db_path = _web_cache_path(ws_root)
    conn = sqlite3.connect(str(db_path))
    try:
        _web_cache_init(conn)

        page_texts: List[Dict[str, str]] = []
        for u in urls[:max_pages]:
            cached = _web_cache_get(conn, u, max_age_days=max_age_days)
            if cached and cached.get("text"):
                page_texts.append({"url": u, "title": cached.get("title", u), "text": cached.get("text", "")})
                continue

            try:
                fetched = _fetch_url_text(u, timeout_s=timeout_s)
                final_url = fetched.get("url", u)
                title = fetched.get("title", final_url)
                text = fetched.get("text", "")
                _web_cache_put(conn, final_url, etag=fetched.get("etag", ""), last_modified=fetched.get("last_modified", ""), title=title, text=text)
                page_texts.append({"url": final_url, "title": title, "text": text})
            except Exception:
                # Ignore fetch failures; we'll return whatever we can.
                continue

        # Build chunk hits
        hits: List[Dict[str, object]] = []
        for page in page_texts:
            chunks = _chunk_for_snippets(page.get("text", ""))
            for ch in chunks:
                sc = _score_chunk(ch, query)
                if sc <= 0:
                    continue
                snip = ch
                if len(snip) > 700:
                    snip = snip[:700] + "…"
                hits.append({
                    "source": f"WEB:{page.get('url','')}",
                    "title": str(page.get("title", "")),
                    "snippet": snip,
                    "score": float(sc),
                })

        # If no chunks matched, still return page title as a weak hit
        if not hits:
            for page in page_texts[:k]:
                hits.append({
                    "source": f"WEB:{page.get('url','')}",
                    "title": str(page.get("title", "")),
                    "snippet": (page.get("text", "") or "")[:700],
                    "score": 0.01,
                })

        # Rank by score
        hits_sorted = sorted(hits, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out: List[dict] = []
        seen = set()
        for h in hits_sorted:
            key = (h.get("source"), h.get("snippet"))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "source": h.get("source"),
                "title": h.get("title"),
                "snippet": h.get("snippet"),
                "score": h.get("score"),
            })
            if len(out) >= k:
                break
        return out
    finally:
        conn.close()
