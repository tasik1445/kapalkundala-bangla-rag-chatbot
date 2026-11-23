import json
import pytest
from types import SimpleNamespace
from src.rag_qa import ABSTAIN, answer_question, verify_response, make_context


def doc(text='নবকুমার শর্ম্মা নামক এক ব্যক্তি এখানে উপস্থিত ছিলেন।'):
    return SimpleNamespace(page_content=text,metadata={
        'volume':'প্রথম খণ্ড','chapter':'অষ্টম পরিচ্ছেদ',
        'source_url':'https://bn.wikisource.org/wiki/test'})


def payload(quote='নবকুমার শর্ম্মা', sid='S1', answer='তাঁর নাম নবকুমার শর্ম্মা।'):
    return json.dumps({'answerable':True,'answer':answer,
                       'evidence':[{'source_id':sid,'evidence_quote':quote}]},ensure_ascii=False)

class Retriever:
    def __init__(self,docs):self.docs=docs
    def invoke(self,q):return self.docs


def test_accept_grounded_answer():
    a=verify_response(payload(),[doc()],'নাম কী?','small_500_100')
    assert a.status=='answered' and a.citations[0].chapter=='অষ্টম পরিচ্ছেদ'

def test_reject_fake_quote():
    a=verify_response(payload(quote='নবকুমার বিজ্ঞানী ছিলেন'),[doc()],'পেশা কী?','small_500_100')
    assert a.status=='not_found' and a.answer==ABSTAIN

def test_reject_fake_source():
    assert verify_response(payload(sid='S9'),[doc()],'Q','small_500_100').status=='not_found'

def test_reject_missing_quote():
    assert verify_response(payload(quote=''),[doc()],'Q','small_500_100').status=='not_found'

def test_reject_empty_answer():
    assert verify_response(payload(answer=''),[doc()],'Q','small_500_100').status=='not_found'

def test_abstain_explicit():
    response=json.dumps({'answerable':False,'answer':'','evidence':[]})
    assert verify_response(response,[doc()],'Q','small_500_100').status=='not_found'

def test_reject_malformed_json():
    assert verify_response('nonsense',[doc()],'Q','small_500_100').status=='not_found'

def test_reject_untrusted_url():
    bad=doc();bad.metadata['source_url']='https://evil.example.com'
    assert verify_response(payload(),[bad],'Q','small_500_100').status=='not_found'

def test_reject_too_short_quote():
    assert verify_response(payload(quote='নাম'),[doc()],'Q','small_500_100').status=='not_found'

def test_context_source_id():
    assert '[S1]' in make_context([doc()])

def test_empty_retrieval_skips_llm():
    def fail(*args):raise AssertionError('LLM must not be invoked')
    assert answer_question('Q',Retriever([]),llm_fn=fail).status=='not_found'

def test_mock_end_to_end():
    a=answer_question('নাম কী?',Retriever([doc()]),llm_fn=lambda q,c:payload())
    assert a.status=='answered' and len(a.citations)==1

def test_query_validation():
    with pytest.raises(ValueError):answer_question('  ',Retriever([doc()]),llm_fn=lambda q,c:payload())
