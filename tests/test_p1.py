import pytest
from src.crawler import BASE, EXPECTED, canonical, chapter_from_href, clean_text, parse_toc, extract_chapter, validate_records


def test_volume_counts():
    assert sum(EXPECTED.values()) == 32


def test_chapter_path_decoded():
    href = BASE + '/প্রথম_খণ্ড/প্রথম_পরিচ্ছেদ'
    assert chapter_from_href(href) == ['প্রথম খণ্ড', 'প্রথম পরিচ্ছেদ']


def test_reject_external():
    assert chapter_from_href('https://example.org/other') is None


def test_reject_outside_book():
    assert chapter_from_href('https://bn.wikisource.org/wiki/অন্য_বই/প্রথম_খণ্ড/প্রথম_পরিচ্ছেদ') is None


def test_canonical_url_removes_query():
    assert '?' not in canonical(BASE + '?a=1#section')


def test_block_external_host():
    with pytest.raises(ValueError):
        canonical('https://evil.example/wiki/fake')


def test_toc_detects_chapters_and_skips_navigation():
    html = f'''<div id="mw-content-text"><div class="mw-parser-output">
    <a href="/wiki/কপালকুণ্ডলা_(বঙ্কিমচন্দ্র_চট্টোপাধ্যায়,_১৮৭০)/প্রথম_খণ্ড/প্রথম_পরিচ্ছেদ">প্রথম খণ্ড — প্রথম পরিচ্ছেদ</a>
    <a href="/wiki/অন্য_বই/প্রথম_খণ্ড/প্রথম_পরিচ্ছেদ">unrelated</a></div></div>'''
    records = parse_toc(html)
    assert len(records) == 1 and records[0]['volume'] == 'প্রথম খণ্ড'


def test_extract_chapter_keeps_bangla_removes_nav():
    html = '<div id="mw-content-text"><div class="mw-parser-output"><div class="prp-pages-output"><p>একটি অনেক বড় বাংলা গল্পের কথা এখানে বর্ণিত হয়েছে। এই অংশটিতে চরিত্র ও ঘটনা এসেছে।</p><p>দ্বিতীয় অংশে আবার নতুন ঘটনা দেখা গেল এবং সংলাপও শুরু হল।</p><div class="ws-noexport">SHOULD_NOT_SHOW</div></div></div></div>'
    txt = extract_chapter(html)
    assert 'একটি' in txt and 'SHOULD_NOT_SHOW' not in txt


def test_nfc_cleaning():
    assert clean_text('বাংলা\u200b\n\n\n বই') == 'বাংলা\n\nবই'


def test_validation_rejects_incomplete():
    assert any('EXPECTED' in p for p in validate_records([]))


def test_validation_detects_duplicate():
    record = {'volume': 'প্রথম খণ্ড', 'chapter_id': 'same', 'source_url': BASE}
    problems = validate_records([record, record])
    assert 'Duplicate chapter identifiers' in problems


def test_detect_short_text():
    row = {'volume':'প্রথম খণ্ড', 'chapter_id':'a', 'source_url': BASE, 'text': 'abc'}
    assert any('suspicious' in x for x in validate_records([row]))
