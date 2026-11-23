"""Regression tests for Groq response compatibility (no live network calls)."""
import json
import sys
from types import SimpleNamespace

import pytest

from src import rag_qa


class FakeClient:
    outputs = []
    calls = []

    def __init__(self, **kwargs):
        pass

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.outputs.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])


@pytest.fixture(autouse=True)
def patch_api(monkeypatch):
    FakeClient.calls = []
    FakeClient.outputs = []
    monkeypatch.setenv('GROQ_API_KEY', 'fake_test_key')
    monkeypatch.setenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    monkeypatch.setitem(sys.modules, 'groq', SimpleNamespace(Groq=FakeClient))
    monkeypatch.setitem(sys.modules, 'dotenv', SimpleNamespace(load_dotenv=lambda *a, **kw: None))


def test_json_text_mode_success():
    payload = {"answerable": False, "answer": "", "evidence": []}
    FakeClient.outputs = [json.dumps(payload)]
    got = rag_qa.groq_answer('test', '[S1] passage')
    assert json.loads(got) == payload
    assert len(FakeClient.calls) == 1
    assert 'response_format' not in FakeClient.calls[0]


def test_invalid_json_retry_then_success():
    FakeClient.outputs = ['Oops no JSON', '{"answerable":false,"answer":"","evidence":[]}']
    got = rag_qa.groq_answer('test', '[S1] passage')
    assert json.loads(got)['answerable'] is False
    assert len(FakeClient.calls) == 2


def test_invalid_json_twice_raises():
    FakeClient.outputs = ['broken', 'still broken']
    with pytest.raises(RuntimeError, match='invalid JSON twice'):
        rag_qa.groq_answer('test', '[S1] passage')
    assert len(FakeClient.calls) == 2
