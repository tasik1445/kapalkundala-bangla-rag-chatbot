# P4 — Grounded Bangla Question Answering

Merge this patch into `D:\Kapalkundala_RAG_P1`, keeping all P1–P3 source and data unchanged.

## Setup

```powershell
cd D:\Kapalkundala_RAG_P1
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-p4.txt
Copy-Item .env.example .env
notepad .env
python -m pytest -q
```

Paste your own Groq API key in `.env`. DO NOT paste it into chat, README, or GitHub. `.env` must be ignored by Git. `GROQ_MODEL` can be changed to a currently supported Groq model.

## Run

```powershell
python -m src.rag_qa --query "নবকুমারের নিবাস কোথায়?" --strategy small_500_100 --k 5
python -m src.rag_qa --query "এই উপন্যাসে মঙ্গল গ্রহের রাজধানীর নাম কী?" --strategy small_500_100 --k 5
```

The first command should be able to answer if a relevant retrieved passage is returned and the LLM supplies a matching evidence quote. The second is an out-of-book question and **must** decline (check the result manually). This is an empirical check, not a guaranteed outcome from tests.

## What this validates

- LangChain retriever loads the **existing** P3 index and verifies P2 source hash and P3 audit.
- Groq LLM receives retrieved passages only, along with instructions not to use external knowledge.
- JSON output includes source IDs and verbatim evidence quotes; a Python validator checks quote matches retrieved text and URL origin.
- Malformed output, fake evidence IDs/quotes and unsupported results fail closed.
- The quote-check is a *partial* guard against hallucination. A correct quote can still fail to logically support the full answer; manual evaluation remains necessary.
- `python -m pytest -q` tests mock responses without API usage, so they cannot prove live answer accuracy.

## Important

This is a terminal RAG pipeline only. Streamlit UI, 10 human-verified expected-answer tests and two-strategy Hit@K comparison belong to P5/P6.

If you see `429 Rate Limit`, stop retrying until Groq quota resets; never expose your API key in screenshots/logs.
