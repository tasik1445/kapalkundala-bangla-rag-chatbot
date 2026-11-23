# Phase P2 — Bengali preprocessing & chunking

## Apply to existing P1 folder

Copy `src/preprocessing.py` to `D:\Kapalkundala_RAG_P1\src\preprocessing.py` and
`tests/test_p2.py` to `D:\Kapalkundala_RAG_P1\tests\test_p2.py`.
Do not overwrite P1 `src/crawler.py`, `data/raw/chapters.json` or the original audit.

```powershell
cd D:\Kapalkundala_RAG_P1
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m src.preprocessing
```

The script **requires P1 full_crawl PASS**, 32 matching TOC IDs, and original
chapter text SHA-256 checks. It normalizes NFC and whitespace without deleting
Bengali diacritics; chunks never span chapter boundaries. Each chunk retains
source URL, chapter and khanda, precise normalized-text character offsets,
chunk hash and deterministic ID.

Two configurations:
- `small_500_100`: 500 characters, 100-character overlap
- `large_900_150`: 900 characters, 150-character overlap

Outputs:
- `data/processed/small_500_100.json`
- `data/processed/large_900_150.json`
- `data/audit/p2_chunk_audit.json`

**Integrity vs semantic quality:** P2 validates exact text coverage and
metadata, not retrieval quality. Hit@3 and Hit@5 require a separate labeled
question set and are measured in P4/P5. Chunk sizes are character-based,
not tokens. P2 uses a standalone deterministic splitter so offline tests work
without new dependencies; P3 will use LangChain's FAISS retriever (and can
compare LangChain text splitting separately if desired).

If original P1 full crawl fails or data has changed, P2 exits nonzero.
