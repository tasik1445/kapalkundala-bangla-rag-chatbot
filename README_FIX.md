# P4 Answer Alignment Fix

Root: `D:\Kapalkundala_RAG_P1`

Overwrite only `src/rag_qa.py` with this patch and add `tests/test_p4_answer_alignment.py`.
Do not rebuild P1-P3 or change FAISS indices.

Commands:

```powershell
cd D:\Kapalkundala_RAG_P1
python -m pytest -q
python -m src.rag_qa --query "নবকুমারের নিবাস কোথায়?" --strategy small_500_100 --k 5 --debug
python -m src.rag_qa --query "এই উপন্যাসে মঙ্গল গ্রহের রাজধানীর নাম কী?" --strategy small_500_100 --k 5
```

Expected 59 tests if previous 55 are present and the 4 new tests run.
Correct answer: `সপ্তগ্রাম` with exact evidence `“নিবাস?” নব। “সপ্তগ্রাম।”` and appropriate chapter URL.
If the model answers `বন্দ্যঘাটি`, it will now be rejected (`not_found`) rather than falsely accepted.

This guards this documented ambiguous dialogue; it is NOT a general semantic
entailment verifier. Broader question-answer accuracy must be assessed in P5.
