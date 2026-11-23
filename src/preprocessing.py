"""P2: chapter-safe Bengali text preprocessing and two controlled chunking policies.

Uses exact character offsets to prevent dropped content; no LLM/API call required.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {"small_500_100": (500, 100), "large_900_150": (900, 150)}
EXPECTED_CHAPTERS = 32


def normalize_bengali(text):
    """NFC, invisible junk removal, and whitespace normalization; keep Bangla marks."""
    if not isinstance(text, str):
        raise TypeError("Chapter text must be a string")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _boundary(text, start, ideal, lower_bound):
    if ideal >= len(text):
        return len(text)
    # Prefer a natural ending within the last 30% of the chunk.
    boundaries = "\n।!?; "
    for i in range(ideal - 1, max(start + 1, lower_bound) - 1, -1):
        if text[i] in boundaries:
            return i + 1
    return ideal


def split_with_offsets(text, size, overlap):
    """Offsets refer to normalized chapter text; chunks exactly equal text[start:end]."""
    if not (isinstance(size, int) and isinstance(overlap, int) and size > 0 and 0 <= overlap < size):
        raise ValueError("Requires size>0 and 0<=overlap<size")
    if not text:
        return []
    results = []
    start = 0
    while start < len(text):
        # For very high overlaps, force full-sized windows to guarantee new content.
        ideal = min(start + size, len(text))
        end = ideal if overlap > size // 2 else _boundary(text, start, ideal, start + int(size * .70))
        if results and end <= results[-1][1]:
            end = min(len(text), max(results[-1][1] + 1, ideal))
        if end <= start:
            raise RuntimeError("Chunker failed to make progress")
        results.append((start, end, text[start:end]))
        if end == len(text):
            break
        new_start = max(start + 1, end - overlap)
        # Avoid beginning in the middle of a word where possible, but not
        # with zero overlap or extreme overlap (these are stress-test cases).
        if overlap and overlap <= size // 2 and new_start > 0 and new_start < len(text):
            while new_start > start + 1 and not text[new_start - 1].isspace() and not text[new_start].isspace():
                new_start -= 1
        if new_start > end or new_start <= start:
            raise RuntimeError("Invalid next chunk start")
        start = new_start
    return results


def validate_coverage(text, chunks):
    if not chunks:
        return False
    if chunks[0][0] != 0 or chunks[-1][1] != len(text):
        return False
    for i, (start, end, content) in enumerate(chunks):
        if not 0 <= start < end <= len(text) or text[start:end] != content:
            return False
        if i and (start > chunks[i-1][1] or end <= chunks[i-1][1]):
            return False
    return True


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_p1(p1, records):
    if p1.get("status") != "PASS" or p1.get("mode") != "full_crawl":
        raise ValueError("P1 full-crawl integrity audit must be PASS")
    if p1.get("downloaded_count") != EXPECTED_CHAPTERS or len(records) != EXPECTED_CHAPTERS:
        raise ValueError("Expected exactly 32 downloaded chapter records")
    ids = [r["chapter_id"] for r in records]
    if len(set(ids)) != EXPECTED_CHAPTERS:
        raise ValueError("Duplicate chapter IDs")
    toc_ids = {r["chapter_id"] for r in p1.get("chapters", [])}
    if set(ids) != toc_ids:
        raise ValueError("Chapter identifiers do not match P1 TOC audit")
    for r in records:
        if not all(k in r for k in ("chapter_id", "volume", "source_url", "display_title", "text", "sha256")):
            raise ValueError("Chapter is missing provenance fields")
        if sha(r["text"]) != r["sha256"]:
            raise ValueError("Source text hash mismatch: " + r["chapter_id"])


def build(records, size, overlap):
    rows, info = [], []
    for chapter_order, record in enumerate(records, start=1):
        normalized = normalize_bengali(record["text"])
        if len(normalized) < 250 or len(re.findall(r"[\u0980-\u09FF]", normalized)) < 80:
            raise ValueError("Suspiciously short Bengali text: " + record["chapter_id"])
        fragments = split_with_offsets(normalized, size, overlap)
        if not validate_coverage(normalized, fragments):
            raise AssertionError("Text coverage check failed: " + record["chapter_id"])
        info.append({"chapter_id":record["chapter_id"],"chapter_order":chapter_order,
                     "normalized_chars":len(normalized), "chunk_count":len(fragments),
                     "original_sha256":record["sha256"],"clean_sha256":sha(normalized),
                     "text_coverage":"PASS"})
        for chunk_number, (start, end, content) in enumerate(fragments, start=1):
            rows.append({"chunk_id":f"c{chapter_order:02d}-{chunk_number:03d}",
                         "chapter_id":record["chapter_id"],"chapter_order":chapter_order,
                         "chunk_number":chunk_number,"volume":record["volume"],
                         "chapter":record["chapter"],"display_title":record["display_title"],
                         "source_url":record["source_url"],"start_char":start,"end_char":end,
                         "text":content,"text_sha256":sha(content)})
    if len({r['chunk_id'] for r in rows}) != len(rows):
        raise AssertionError('Duplicate chunk IDs')
    return rows, info


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def run(root=ROOT):
    raw=root/'data/raw/chapters.json'
    audit=root/'data/audit/p1_integrity.json'
    if not raw.is_file() or not audit.is_file():
        raise FileNotFoundError("Run python -m src.crawler successfully first; missing P1 files")
    records=json.loads(raw.read_text(encoding='utf-8'))
    p1=json.loads(audit.read_text(encoding='utf-8'))
    check_p1(p1,records)
    results={}
    generated={}
    for name,(size,overlap) in STRATEGIES.items():
        chunks, chapters=build(records,size,overlap)
        results[name]={"chunk_size":size,"overlap":overlap,"chapters":len(chapters),
                       "chunks":len(chunks),"shortest_chunk_chars":min(len(x['text']) for x in chunks),
                       "longest_chunk_chars":max(len(x['text']) for x in chunks),
                       "mean_chunk_chars":round(sum(len(x['text']) for x in chunks)/len(chunks),2),
                       "coverage":"PASS","per_chapter":chapters}
        generated[name]=chunks
    # Stage output only after both strategies pass. A manifest helps P3 verify provenance.
    out=root/'data/processed'
    for name,chunks in generated.items():
        write_json(out/f'{name}.json',chunks)
        print(f'{name}: {len(chunks)} chunks from 32 chapters — coverage PASS')
    report={"phase":"P2","status":"PASS","book":"কপালকুণ্ডলা", "source_p1_status":"PASS",
            "generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "bonus_note":"No retrieval hit-rate measured yet; compare strategies on fixed Q/A in P4/P5",
            "strategies":results}
    write_json(root/'data/audit/p2_chunk_audit.json',report)
    print('FINAL P2 AUDIT: PASS — both chunking strategies cover 32/32 chapters')
    return report


def main():
    try:
        run()
    except (ValueError, FileNotFoundError, KeyError, AssertionError, json.JSONDecodeError) as exc:
        print(f'FINAL P2 AUDIT: FAIL — {type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
