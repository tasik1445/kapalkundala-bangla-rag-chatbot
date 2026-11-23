# Phase P3 — Multilingual E5 + FAISS + LangChain

**Prerequisite:** `python -m src.preprocessing` must have printed `FINAL P2 AUDIT: PASS`.

Copy `src/vector_index.py`, `tests/test_p3.py`, and `requirements-p3.txt` to the corresponding paths in the existing `D:\Kapalkundala_RAG_P1` project. Do not overwrite any P1/P2 file or delete the corpus.

## Windows PowerShell

```powershell
cd D:\Kapalkundala_RAG_P1
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-p3.txt
python -m pip check
python -m pytest -q
python -m src.vector_index
python -m src.vector_index --query "নবকুমার কে?" --strategy small_500_100 --k 3
python -m src.vector_index --query "নবকুমার কে?" --strategy large_900_150 --k 3
```

**Note:** First run downloads model weights from Hugging Face (~1GB and dependencies); requires internet and available disk. CPU works; GPU optionally `--device cuda` if PyTorch CUDA is configured. No Groq key is needed in P3. Two indices use the same model with the same query/document prefixes for later controlled bonus comparison.

### Expected artifact layout

```text
data/
  audit/p3_vector_audit.json
  vectorstores/
    small_500_100/vectors.faiss
                  /documents.json
                  /manifest.json
    large_900_150/vectors.faiss
                  /documents.json
                  /manifest.json
```

Expected counts for your current P2 input: **391** vectors (small) and **214** vectors (large). Final audit should print `FINAL P3 AUDIT: PASS — both FAISS indices built and verified`.

### Security and data integrity

- Validate P2 audit and all chunk text SHA-256 before embedding.
- Preserve source URLs, volume/chapter labels and chunk offsets as LangChain Document metadata.
- Use `passage: ` on document texts, `query: ` on questions, L2 normalization, and FAISS inner product (cosine equivalent).
- Do **not** load untrusted FAISS index files. The loader verifies SHA-256 before loading and never uses pickle serialization.
- P3 is index construction / smoke retrieval only. No retrieval quality or final answer accuracy is claimed before P4/P5 evaluation.

If PowerShell displays mangled Bangla inside `Get-Content`, the JSON may still be valid UTF-8; inspect in VS Code. To display text in terminal, use `Get-Content -Encoding UTF8`.
