"""P3: build and validate two independent FAISS+LangChain Bengali book indices.

Source of truth: P2 chunk JSON and P2 audit; never silently recrawl or re-chunk.
No unsafe pickle deserialization. E5 requires 'passage:' and 'query:' prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = 'intfloat/multilingual-e5-base'
STRATEGIES = ('small_500_100', 'large_900_150')
EXPECTED_COUNTS = None  # counts read from immutable P2 audit, not hard-coded


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def validate_p2(root: Path = ROOT):
    audit = _load_json(root / 'data/audit/p2_chunk_audit.json')
    if audit.get('phase') != 'P2' or audit.get('status') != 'PASS' or audit.get('source_p1_status') != 'PASS':
        raise ValueError('P2 audit must be PASS with validated P1 source')
    result = {}
    for strategy in STRATEGIES:
        spec = audit.get('strategies', {}).get(strategy, {})
        if spec.get('coverage') != 'PASS' or spec.get('chapters') != 32:
            raise ValueError('P2 strategy not validated: ' + strategy)
        path = root / 'data/processed' / f'{strategy}.json'
        chunks = _load_json(path)
        if not isinstance(chunks, list) or len(chunks) != spec.get('chunks'):
            raise ValueError('P2 chunk count mismatch: ' + strategy)
        ids = [c['chunk_id'] for c in chunks]
        if len(ids) != len(set(ids)) or len({c['chapter_id'] for c in chunks}) != 32:
            raise ValueError('Duplicate chunk IDs or missing chapters: ' + strategy)
        for c in chunks:
            required = ('chunk_id','chapter_id','chapter_order','chunk_number','volume',
                        'chapter','display_title','source_url','text','text_sha256',
                        'start_char','end_char')
            if not all(key in c for key in required):
                raise ValueError('Missing chunk metadata: ' + c.get('chunk_id','?'))
            if not c['text'] or _sha_bytes(c['text'].encode('utf-8')) != c['text_sha256']:
                raise ValueError('Chunk text SHA mismatch: ' + c['chunk_id'])
            if not str(c['source_url']).startswith('https://bn.wikisource.org/wiki/'):
                raise ValueError('Invalid source URL: ' + c['chunk_id'])
        result[strategy] = (chunks, _sha_file(path))
    return result


try:
    from langchain_core.embeddings import Embeddings as _LangChainEmbeddings
except ImportError:  # permits dependency-free validation tests before P3 install
    class _LangChainEmbeddings:
        pass


class E5Embeddings(_LangChainEmbeddings):
    """LangChain Embeddings protocol, lazy loaded to avoid a model download during tests."""
    def __init__(self, model_name=MODEL_NAME, batch_size=16, device='cpu'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get().encode(['passage: ' + s for s in texts],
                                     batch_size=self.batch_size,
                                     normalize_embeddings=True,
                                     show_progress_bar=True,
                                     convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._get().encode('query: ' + text, normalize_embeddings=True,
                                    convert_to_numpy=True)
        return vector.tolist()


def _validate_vectors(vectors, n):
    import numpy as np
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or len(arr) != n or arr.shape[1] < 2:
        raise ValueError('Embedding count or dimensionality mismatch')
    if not np.isfinite(arr).all():
        raise ValueError('Embeddings contain NaN/Inf')
    lengths = np.linalg.norm(arr, axis=1)
    if np.any(lengths < 1e-7):
        raise ValueError('Zero embedding vector')
    arr = np.ascontiguousarray(arr / lengths[:, None], dtype=np.float32)
    return arr


def build_index(chunks: list[dict], embedding, output_dir: Path, model_name: str, chunks_sha: str):
    import faiss
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.utils import DistanceStrategy

    if not chunks:
        raise ValueError('No chunks to index')
    texts = [c['text'] for c in chunks]
    arr = _validate_vectors(embedding.embed_documents(texts), len(texts))
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    docs = {}
    ids = {}
    for i,c in enumerate(chunks):
        chunk_id=c['chunk_id']
        if chunk_id in docs:
            raise ValueError('Duplicate chunk ID: '+chunk_id)
        metadata = {key:val for key,val in c.items() if key != 'text'}
        docs[chunk_id] = Document(page_content=c['text'], metadata=metadata)
        ids[i] = chunk_id
    store = FAISS(embedding_function=embedding,index=index,
                 docstore=InMemoryDocstore(docs),index_to_docstore_id=ids,
                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    output_dir.mkdir(parents=True,exist_ok=True)
    faiss_path=output_dir/'vectors.faiss'
    faiss.write_index(index,str(faiss_path))
    _write_json(output_dir/'documents.json', [{'id':ids[i], 'page_content':docs[ids[i]].page_content,
                                                'metadata':docs[ids[i]].metadata}
                                              for i in range(len(ids))])
    manifest={'status':'PASS','model':model_name,'e5_prefix':{'document':'passage: ','query':'query: '},
              'normalized':True,'similarity':'cosine via normalized inner product',
              'dimension':int(arr.shape[1]),'vectors':int(index.ntotal),'chapters':32,
              'chunk_source_sha256':chunks_sha,'index_sha256':_sha_file(faiss_path),
              'documents_sha256':_sha_file(output_dir/'documents.json')}
    _write_json(output_dir/'manifest.json',manifest)
    return store,manifest


def load_index(index_dir: Path, embedding, expected_model: str = MODEL_NAME):
    import faiss
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.utils import DistanceStrategy

    manifest=_load_json(index_dir/'manifest.json')
    if manifest.get('status')!='PASS' or manifest.get('model')!=expected_model:
        raise ValueError('Index manifest/model mismatch')
    if _sha_file(index_dir/'vectors.faiss')!=manifest['index_sha256'] or _sha_file(index_dir/'documents.json')!=manifest['documents_sha256']:
        raise ValueError('Stored FAISS or document hash mismatch')
    rows=_load_json(index_dir/'documents.json')
    if len(rows)!=manifest['vectors'] or len({row['id'] for row in rows})!=len(rows):
        raise ValueError('Saved documents count or ID mismatch')
    index=faiss.read_index(str(index_dir/'vectors.faiss'))
    if index.ntotal!=len(rows) or index.d!=manifest['dimension']:
        raise ValueError('Index dimension/count mismatch')
    docs={row['id']:Document(page_content=row['page_content'],metadata=row['metadata']) for row in rows}
    store=FAISS(embedding_function=embedding,index=index,
               docstore=InMemoryDocstore(docs),
               index_to_docstore_id={i:row['id'] for i,row in enumerate(rows)},
               distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    return store


def run(root=ROOT, model_name=MODEL_NAME, device='cpu', batch_size=16):
    source=validate_p2(root)
    embedder=E5Embeddings(model_name=model_name,device=device,batch_size=batch_size)
    result={}
    for strategy in STRATEGIES:
        chunks, digest=source[strategy]
        print(f'[{strategy}] embedding {len(chunks)} Bangla chunks ...',flush=True)
        folder=root/'data/vectorstores'/strategy
        _, manifest=build_index(chunks,embedder,folder,model_name,digest)
        reloaded=load_index(folder,embedder,model_name)
        if reloaded.index.ntotal!=len(chunks):
            raise AssertionError('Reload failed')
        result[strategy]=manifest
        print(f'[{strategy}] FAISS saved/reloaded: {manifest["vectors"]} vectors, dimension={manifest["dimension"]}',flush=True)
    audit={'phase':'P3','status':'PASS','embedding_model':model_name,
           'source_p2_status':'PASS','generated_at_utc':datetime.now(timezone.utc).isoformat(),
           'strategies':result,'note':'P3 verifies indexing only. Retrieval hit rates not evaluated yet.'}
    _write_json(root/'data/audit/p3_vector_audit.json',audit)
    print('FINAL P3 AUDIT: PASS — both FAISS indices built and verified',flush=True)
    return audit


def query(root=ROOT,strategy='small_500_100',question='নবকুমার কে?',k=3,model_name=MODEL_NAME,device='cpu'):
    if strategy not in STRATEGIES or not 1<=k<=10:
        raise ValueError('Invalid strategy or k')
    validated=validate_p2(root)
    manifest=_load_json(root/'data/vectorstores'/strategy/'manifest.json')
    if manifest['chunk_source_sha256']!=validated[strategy][1]:
        raise ValueError('Stored index is out of date with P2 data; rebuild P3')
    embedder=E5Embeddings(model_name=model_name,device=device)
    store=load_index(root/'data/vectorstores'/strategy,embedder,expected_model=model_name)
    retriever=store.as_retriever(search_kwargs={'k':k})
    docs=retriever.invoke(question)
    for i,doc in enumerate(docs,1):
        m=doc.metadata
        print(f'[{i}] {m["volume"]} / {m["chapter"]} | {m["source_url"]}')
        print('    '+doc.page_content[:180].replace('\n',' '))
    return docs


def main():
    parser=argparse.ArgumentParser(description='P3 multilingual E5 + FAISS index builder')
    parser.add_argument('--model',default=MODEL_NAME)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--query',default=None)
    parser.add_argument('--strategy',choices=STRATEGIES,default=STRATEGIES[0])
    parser.add_argument('--k',type=int,default=3)
    args=parser.parse_args()
    try:
        if args.query:
            query(strategy=args.strategy,question=args.query,k=args.k,model_name=args.model,device=args.device)
        else:
            run(model_name=args.model,device=args.device,batch_size=args.batch_size)
    except Exception as err:
        print(f'P3 FAILED: {type(err).__name__}: {err}',flush=True)
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
