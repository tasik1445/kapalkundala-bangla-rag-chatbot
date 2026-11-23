import json
from types import SimpleNamespace
from src.rag_qa import explain_validation, answer_question


def doc(text='নব। “নবকুমার শর্ম্মা।” অধি। “নিবাস?” নব। “সপ্তগ্রাম।”'):
    return SimpleNamespace(page_content=text, metadata={'volume':'প্রথম খণ্ড','chapter':'অষ্টম পরিচ্ছেদ','source_url':'https://bn.wikisource.org/wiki/test'})

class Retriever:
    def invoke(self,q):
        return [doc()]

def test_diagnostic_distinguishes_false_model(capsys):
    a=answer_question('নবকুমারের নিবাস কোথায়?',Retriever(),llm_fn=lambda q,c:'{"answerable":false,"answer":"","evidence":[]}',debug=True)
    assert a.status=='not_found'
    assert 'model_abstained' in capsys.readouterr().out

def test_diagnostic_identifies_wrong_quote():
    raw=json.dumps({'answerable':True,'answer':'সপ্তগ্রাম','evidence':[{'source_id':'S1','evidence_quote':'ভুল বানানো প্রমাণের দীর্ঘ টেক্সট'}]},ensure_ascii=False)
    assert explain_validation(raw,[doc()])=='quote_not_exactly_in_retrieved_text'

def test_diagnostic_valid_evidence():
    raw=json.dumps({'answerable':True,'answer':'সপ্তগ্রাম','evidence':[{'source_id':'S1','evidence_quote':'“নিবাস?” নব। “সপ্তগ্রাম।”'}]},ensure_ascii=False)
    assert explain_validation(raw,[doc()])=='evidence_structure_valid'
    a=answer_question('নিবাস?',Retriever(),llm_fn=lambda q,c:raw,debug=True)
    assert a.status=='answered' and a.citations[0].chapter=='অষ্টম পরিচ্ছেদ'

def test_invalid_json_reason():
    assert explain_validation('not JSON',[doc()]).startswith('invalid_json')
