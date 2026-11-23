# Kapalkundala RAG — Phase P1: Full Book Ingestion

> **Status:** P1 code and offline parser tests available. A full live crawl on the user's Windows machine is required before declaring the book ingested. This ZIP is NOT a completed RAG chatbot.

**Book:** কপালকুণ্ডলা (বঙ্কিমচন্দ্র চট্টোপাধ্যায়, ১৮৭০)

**Source:** https://bn.wikisource.org/wiki/কপালকুণ্ডলা_(বঙ্কিমচন্দ্র_চট্টোপাধ্যায়,_১৮৭০)

**TOC audit from web:** 4 parts, 9 + 6 + 7 + 10 = **32 expected chapters**. The Python crawler independently extracts all TOC chapter URLs, fetches each, retains titles/URLs, verifies per-part counts and nontrivial Bengali text, writes a fail-closed audit.

## Windows VS Code

```powershell
cd D:\Kapalkundala_RAG_P1
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.crawler --dry-run
python -m src.crawler
```

Expected: `Discovered 32/32 chapter links`, then `FINAL P1 AUDIT: PASS 32/32 downloaded` **only after successful real crawl**.

## Artifacts after P1 successful crawl

- `data/audit/p1_integrity.json`: audit PASS/FAIL and complete TOC URL manifest
- `data/raw/chapters.json`: one record per chapter with book section, source URL, text, SHA-256 hash
- `data/raw/book_full.txt`: readable merged book text with per-chapter source URLs

**If audit FAIL:** Do not train embeddings or claim the book is complete. Review errors, HTML extraction and missing chapters. The crawler does not overwrite validated corpus on a failed run.

## Next phases (not included yet)

P2 normalization/chunking and comparison; P3 multilingual embeddings/FAISS; P4 LangChain RAG citations and abstention; P5 10-question testing and Hit@k comparison; P6 Streamlit demo, README and video.

## Source ethics

Chapter text comes from an acknowledged public-domain book on Bengali Wikisource. Respect service limits, include direct chapter citations, and avoid rapid bulk requests. The crawler defaults to 1.25 seconds between chapter-page requests, with retries on transient network errors.

## Book registration

The student must check and register the unique title in the course's shared sheet; this project does not register it.
