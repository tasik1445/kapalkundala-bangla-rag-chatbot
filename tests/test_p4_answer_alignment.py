import json
from types import SimpleNamespace
from src.rag_qa import verify_response

PASSAGE = 'অধি। “নিবাস?” নব। “সপ্তগ্রাম।” অধি। “আপনারা কোন্ গাঁই।” নব। “বন্দ্যঘাটি।”'
DOC = SimpleNamespace(page_content=PASSAGE, metadata={
    'volume': 'প্রথম খণ্ড',
    'chapter': 'অষ্টম পরিচ্ছেদ',
    'source_url': 'https://bn.wikisource.org/wiki/sample',
})


def response(answer, quote):
    return json.dumps({'answerable':True, 'answer':answer, 'evidence':[
        {'source_id':'S1','evidence_quote': quote}]},ensure_ascii=False)


def test_wrong_relation_rejected_even_when_quote_is_real():
    r=verify_response(response('বন্দ্যঘাটি', PASSAGE), [DOC], 'নবকুমারের নিবাস কোথায়?', 'small_500_100')
    assert r.status == 'not_found'


def test_correct_residence_accepted_with_exact_qa_pair():
    r=verify_response(response('সপ্তগ্রাম', '“নিবাস?” নব। “সপ্তগ্রাম।”'), [DOC], 'নবকুমারের নিবাস কোথায়?', 'small_500_100')
    assert r.status == 'answered'
    assert r.answer == 'সপ্তগ্রাম'


def test_village_quote_alone_not_enough_for_residence():
    r=verify_response(response('সপ্তগ্রাম', '“আপনারা কোন্ গাঁই।” নব। “বন্দ্যঘাটি।”'), [DOC], 'নবকুমারের নিবাস কোথায়?', 'small_500_100')
    assert r.status == 'not_found'


def test_other_questions_remain_supported():
    r=verify_response(response('বন্দ্যঘাটি', '“আপনারা কোন্ গাঁই।” নব। “বন্দ্যঘাটি।”'), [DOC], 'নবকুমার কোন্ গাঁয়ের?', 'small_500_100')
    assert r.status == 'answered'
