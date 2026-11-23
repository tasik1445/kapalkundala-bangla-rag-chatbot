import hashlib
import json
import pytest
from src.preprocessing import (normalize_bengali, split_with_offsets, validate_coverage,
                               build, check_p1, run, sha, STRATEGIES)


def sample_records():
    records=[]
    for i in range(32):
        text=("কপালকুণ্ডলা বনের পথে যাত্রা করিলেন। নবকুমার তখন প্রশ্ন করিলেন। " * 12).strip()
        records.append({"chapter_id":f"volume-{i}/chapter-{i}","volume":"প্রথম খণ্ড",
                        "chapter":str(i),"display_title":f"পরিচ্ছেদ {i}",
                        "source_url":f"https://bn.wikisource.org/wiki/test/{i}",
                        "text":text,"sha256":sha(text)})
    return records


def test_nfc():
    assert normalize_bengali(' আ\u200bমি  বাংলা\n\n\n লিখি ') == 'আমি বাংলা\n\nলিখি'


@pytest.mark.parametrize('size,overlap', [(500,100),(900,150),(30,0),(10,9)])
def test_coverage(size,overlap):
    text='বাংলাদেশের ইতিহাস। আমাদের সাহিত্য পড়ি। '*80
    chunks=split_with_offsets(text,size,overlap)
    assert validate_coverage(text,chunks)
    assert all(b-a<=size for a,b,_ in chunks)


def test_short():
    assert split_with_offsets('বাংলা',500,100)==[(0,5,'বাংলা')]


@pytest.mark.parametrize('size,overlap',[(0,0),(10,10),(10,-1)])
def test_invalid_size(size,overlap):
    with pytest.raises(ValueError):split_with_offsets('abc',size,overlap)


def test_empty():
    assert split_with_offsets('',500,100)==[]
    assert not validate_coverage('',[])


def test_changed_hash_rejected():
    records=sample_records()
    records[0]['text']='changed'
    audit={'status':'PASS','mode':'full_crawl','downloaded_count':32,
           'chapters':[{'chapter_id':r['chapter_id']} for r in records]}
    with pytest.raises(ValueError,match='hash mismatch'):check_p1(audit,records)


def test_not_pass_rejected():
    records=sample_records()
    with pytest.raises(ValueError):check_p1({'status':'FAIL','mode':'full_crawl'},records)


def test_build_chapter_boundary_and_provenance():
    records=sample_records()
    rows,stats=build(records,500,100)
    assert len(stats)==32
    assert len({r['chapter_id'] for r in rows})==32
    assert all(r['source_url'].startswith('https://bn.wikisource.org/wiki/') for r in rows)
    assert len({r['chunk_id'] for r in rows})==len(rows)


def test_end_to_end(tmp_path):
    records=sample_records()
    (tmp_path/'data/raw').mkdir(parents=True)
    (tmp_path/'data/audit').mkdir(parents=True)
    (tmp_path/'data/raw/chapters.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    audit={'status':'PASS','mode':'full_crawl','downloaded_count':32,
           'chapters':[{'chapter_id':r['chapter_id']} for r in records]}
    (tmp_path/'data/audit/p1_integrity.json').write_text(json.dumps(audit),encoding='utf-8')
    result=run(tmp_path)
    assert result['status']=='PASS'
    assert len(list((tmp_path/'data/processed').glob('*.json')))==2
    assert all(row['coverage']=='PASS' for row in result['strategies'].values())
