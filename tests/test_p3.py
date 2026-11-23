import hashlib
import json
import pytest
from src.vector_index import E5Embeddings, STRATEGIES, _validate_vectors, validate_p2, _sha_file


def test_strategies():
    assert STRATEGIES == ('small_500_100','large_900_150')


def test_e5_prefixes_without_network():
    class FakeModel:
        def __init__(self): self.seen=[]
        def encode(self,x,**kw):
            self.seen.append(x)
            if isinstance(x,list): return type('A',(),{'tolist':lambda self:[[1.0,0.0] for _ in x]})()
            return type('A',(),{'tolist':lambda self:[0.0,1.0]})()
    emb=E5Embeddings()
    fake=FakeModel()
    emb._model=fake
    assert emb.embed_documents(['বাংলা','বই'])==[[1.,0.],[1.,0.]]
    assert emb.embed_query('প্রশ্ন')==[0.,1.]
    assert fake.seen==[['passage: বাংলা','passage: বই'],'query: প্রশ্ন']


def test_embedding_zero_and_nan():
    with pytest.raises(ValueError):_validate_vectors([[0,0]],1)
    with pytest.raises(ValueError):_validate_vectors([[1,float('nan')]],1)
    with pytest.raises(ValueError):_validate_vectors([[1,2]],2)


def test_embedding_normalization():
    import numpy as np
    a=_validate_vectors([[3,4],[5,12]],2)
    assert np.allclose(np.linalg.norm(a,axis=1),[1,1])


def test_fail_p2_missing(tmp_path):
    with pytest.raises(FileNotFoundError):validate_p2(tmp_path)


def make_p2(tmp_path):
    audit={'phase':'P2','status':'PASS','source_p1_status':'PASS','strategies':{}}
    for name in STRATEGIES:
        chunks=[]
        for j in range(32):
            txt='বাংলা পাঠ্য '
            chunks.append({'chunk_id':f'c{j+1:02d}-001','chapter_id':f'খণ্ড/পরিচ্ছেদ {j}',
              'chapter_order':j+1,'chunk_number':1,'volume':'খণ্ড','chapter':f'পরিচ্ছেদ {j}',
              'display_title':f'পরিচ্ছেদ {j}','source_url':'https://bn.wikisource.org/wiki/test',
              'text':txt,'text_sha256':hashlib.sha256(txt.encode()).hexdigest(),
              'start_char':0,'end_char':len(txt)})
        d=tmp_path/'data/processed';d.mkdir(parents=True,exist_ok=True)
        (d/f'{name}.json').write_text(json.dumps(chunks,ensure_ascii=False),encoding='utf-8')
        audit['strategies'][name]={'coverage':'PASS','chapters':32,'chunks':32}
    d=tmp_path/'data/audit';d.mkdir(parents=True,exist_ok=True)
    (d/'p2_chunk_audit.json').write_text(json.dumps(audit),encoding='utf-8')


def test_validate_p2_with_sample(tmp_path):
    make_p2(tmp_path)
    result=validate_p2(tmp_path)
    assert all(len(v[0])==32 for v in result.values())


def test_detect_tamper(tmp_path):
    make_p2(tmp_path)
    f=tmp_path/'data/processed/small_500_100.json'
    s=json.loads(f.read_text(encoding='utf-8'))
    s[0]['text']='changed'
    f.write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(ValueError,match='SHA mismatch'):validate_p2(tmp_path)


def test_detect_count_mismatch(tmp_path):
    make_p2(tmp_path)
    f=tmp_path/'data/processed/small_500_100.json'
    s=json.loads(f.read_text(encoding='utf-8'))
    s.pop()
    f.write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(ValueError,match='count mismatch'):validate_p2(tmp_path)


def test_index_roundtrip_if_installed(tmp_path):
    pytest.importorskip('faiss')
    pytest.importorskip('langchain_community')
    from src.vector_index import build_index,load_index
    from langchain_core.embeddings import Embeddings
    make_p2(tmp_path)
    chunks=validate_p2(tmp_path)['small_500_100'][0]
    class DemoEmbedding(Embeddings):
        def embed_documents(self,texts):return [[1.0,0.0] for _ in texts]
        def embed_query(self,text):return [1.0,0.0]
    emb=DemoEmbedding()
    folder=tmp_path/'index'
    store,m=build_index(chunks,emb,folder,'test/embed','abc')
    assert store.index.ntotal==32
    reloaded=load_index(folder,emb,'test/embed')
    assert len(reloaded.as_retriever(search_kwargs={'k':2}).invoke('test'))==2
    assert m['vectors']==32
