"""Crawl a public-domain Bengali Wikisource book by its actual chapter links.

Run: python -m src.crawler --dry-run  (TOC only)
     python -m src.crawler            (all chapter pages)

Fail-closed: only writes validated complete corpus and PASS audit when all expected chapters pass.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://bn.wikisource.org/wiki/কপালকুণ্ডলা_(বঙ্কিমচন্দ্র_চট্টোপাধ্যায়,_১৮৭০)'
EXPECTED = {'প্রথম খণ্ড': 9, 'দ্বিতীয় খণ্ড': 6, 'তৃতীয় খণ্ড': 7, 'চতুর্থ খণ্ড': 10}
EXPECTED_TOTAL = sum(EXPECTED.values())
HEADERS = {'User-Agent': 'KapalkundalaRAGStudentProject/1.0 (educational book ingestion; polite requests)', 'Accept-Language': 'bn,en;q=0.6'}
DELETE_SELECTORS = ', '.join([
    'script','style','noscript','sup.reference','.mw-editsection','.mw-cite-backlink',
    '.mw-references-wrap','.reflist','.reference','.references','.toc','.mw-parser-output .navbox',
    '.ws-noexport','.noprint','.metadata','.mw-empty-elt','.printfooter',
    '.mw-headline .mw-editsection','.mw-collapsible','.mw-collapsed',
    '.catlinks','.mw-authority-control','.mw-indicators','.mw-file-description',
    '.ws-summary','.header-container','.ws-header','.ws-footer',
    '.mw-cite-error','.mw-content-ltr .license',
])


def clean_text(value):
    value = unicodedata.normalize('NFC', value).replace('\u200b', '').replace('\ufeff', '').replace('\xa0', ' ')
    value = re.sub(r'[ \t]+', ' ', value)
    value = re.sub(r' *\n *', '\n', value)
    return re.sub(r'\n{3,}', '\n\n', value).strip()


def canonical(url):
    parts = urlsplit(url)
    if parts.hostname != 'bn.wikisource.org' or not parts.path.startswith('/wiki/'):
        raise ValueError('Unsafe or irrelevant URL: ' + url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def chapter_from_href(href):
    path = unquote(urlsplit(href).path)
    if not path.startswith('/wiki/'):
        return None
    title = path.removeprefix('/wiki/').replace('_', ' ')
    prefix = unquote(urlsplit(BASE).path).removeprefix('/wiki/').replace('_', ' ')
    if not title.startswith(prefix + '/'):
        return None
    subsections = title[len(prefix)+1:].split('/')
    if len(subsections) != 2 or 'পরিচ্ছেদ' not in subsections[1] or subsections[0] not in EXPECTED:
        return None
    return subsections


def parse_toc(html, origin=BASE):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.select_one('#mw-content-text .mw-parser-output') or soup.select_one('.mw-parser-output')
    if not content:
        raise ValueError('TOC parser could not locate Wiki article content')
    found = {}
    for anchor in content.select('a[href]'):
        href = anchor.get('href', '')
        full_url = urljoin(origin, href)
        parts = chapter_from_href(full_url)
        if not parts:
            continue
        volume, chapter = parts
        title = clean_text(anchor.get_text(' ', strip=True)) or (volume + ' — ' + chapter)
        key = volume + '/' + chapter
        record = {'chapter_id': key, 'volume': volume, 'chapter': chapter,
                  'display_title': title, 'source_url': canonical(full_url)}
        if key in found and found[key]['source_url'] != record['source_url']:
            raise ValueError('Conflicting chapter URLs for ' + key)
        found[key] = record
    return list(found.values())


def extract_chapter(html):
    soup = BeautifulSoup(html, 'html.parser')
    main = soup.select_one('#mw-content-text .mw-parser-output') or soup.select_one('.mw-parser-output')
    if not main:
        raise ValueError('Chapter parser could not locate Wiki article content')
    preferred = main.select_one('.prp-pages-output') or main.select_one('.mw-parser-output') or main
    # Avoid accidentally stripping the actual page text; work on parsed HTML copy.
    body = BeautifulSoup(str(preferred), 'html.parser')
    for tag in body.select(DELETE_SELECTORS):
        tag.decompose()
    # Chapter text is commonly rendered as multiple paragraphs inside per-page wrappers.
    paragraphs = [clean_text(p.get_text(' ', strip=True)) for p in body.find_all('p')]
    paragraphs = [p for p in paragraphs if len(p) >= 10]
    if len(' '.join(paragraphs)) < 200:
        # Some Wikisource transclusions use divs/br rather than paragraphs.
        raw = body.get_text('\n', strip=True)
        text = clean_text(raw)
    else:
        text = clean_text('\n\n'.join(paragraphs))
    return text


def validate_records(records, required=EXPECTED, min_chars=250):
    problems = []
    counts = Counter(x['volume'] for x in records)
    if len(records) != sum(required.values()):
        problems.append(f'EXPECTED {sum(required.values())} chapters, GOT {len(records)}')
    for group, target in required.items():
        if counts[group] != target:
            problems.append(f'{group}: expected {target}, got {counts[group]}')
    ids = [x['chapter_id'] for x in records]
    if len(ids) != len(set(ids)):
        problems.append('Duplicate chapter identifiers')
    urls = [x['source_url'] for x in records]
    if len(urls) != len(set(urls)):
        problems.append('Duplicate chapter URLs')
    for x in records:
        if 'text' in x:
            n = len(x['text'])
            bn_count = len(re.findall(r'[\u0980-\u09FF]', x['text']))
            if n < min_chars or bn_count < 80:
                problems.append(f'{x["chapter_id"]}: suspicious content length {n}, Bengali chars {bn_count}')
    return problems


def download(url, client, attempts=3):
    for attempt in range(attempts):
        try:
            response = client.get(url, headers=HEADERS, timeout=(12, 35))
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def audit_and_save(toc, records, errors, dry_run=False):
    issues = validate_records(toc)
    if not dry_run:
        issues += validate_records(records)
        issues.extend(errors)
    report = {
        'phase': 'P1', 'book': 'কপালকুণ্ডলা', 'edition': '১৮৭০',
        'source_url': BASE, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'expected_count': EXPECTED_TOTAL, 'discovered_count': len(toc),
        'downloaded_count': len(records), 'per_volume': dict(Counter(x['volume'] for x in toc)),
        'mode': 'toc_only' if dry_run else 'full_crawl',
        'status': ('DISCOVERY_PASS' if not issues else 'DISCOVERY_FAIL') if dry_run else ('PASS' if not issues else 'FAIL'),
        'issues': issues,
        'chapters': [{k: r[k] for k in ('chapter_id', 'source_url', 'volume', 'chapter')} for r in toc],
    }
    save_json(ROOT / 'data/audit/p1_integrity.json', report)
    if dry_run:
        return report
    if not issues:
        save_json(ROOT / 'data/raw/chapters.json', records)
        lines = ['# কপালকুণ্ডলা — সম্পূর্ণ পরিচ্ছেদ পাঠ\n']
        for r in records:
            lines.append(f'\n## {r["display_title"]}\n\nSource: {r["source_url"]}\n\n{r["text"]}\n')
        (ROOT / 'data/raw/book_full.txt').write_text('\n'.join(lines), encoding='utf-8')
    return report


def run(dry_run=False, delay=1.25):
    with requests.Session() as client:
        print('Fetching TOC from Wikisource ...', flush=True)
        toc_html = download(BASE, client)
        toc = parse_toc(toc_html)
        print(f'Discovered {len(toc)}/{EXPECTED_TOTAL} chapter links', flush=True)
        discovery_issues = validate_records(toc)
        if dry_run or discovery_issues:
            report = audit_and_save(toc, [], [], dry_run=True)
            print('TOC audit:', report['status'], report['issues'], flush=True)
            return 0 if not report['issues'] else 1
        records, errors = [], []
        for i, item in enumerate(toc, 1):
            print(f'[{i}/{len(toc)}] {item["display_title"]}', flush=True)
            try:
                html = download(item['source_url'], client)
                text = extract_chapter(html)
                row = dict(item)
                row['text'] = text
                row['characters'] = len(text)
                row['sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
                records.append(row)
            except (requests.RequestException, ValueError) as exc:
                errors.append(f'{item["chapter_id"]}: {type(exc).__name__}: {exc}')
            time.sleep(delay)
        report = audit_and_save(toc, records, errors)
        print('FINAL P1 AUDIT:', report['status'], f'{len(records)}/{EXPECTED_TOTAL} downloaded', flush=True)
        if report['issues']:
            print('\n'.join(report['issues']), flush=True)
        return 0 if report['status'] == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='Discover all chapter links without downloading chapter text')
    ap.add_argument('--delay', type=float, default=1.25, help='Delay between chapter downloads in seconds')
    args = ap.parse_args()
    if args.delay < 0.5:
        ap.error('Use at least 0.5 seconds between requests')
    try:
        return run(dry_run=args.dry_run, delay=args.delay)
    except (requests.RequestException, ValueError) as exc:
        print('CRAWL FAILED:', type(exc).__name__, exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
