"""Stage B (Mode A): inline image-needle vision read-path.

For inline attack classes whose forged value rides a ``long_context_documents`` entry
(rag_resolve_weights -> SPLIT_EVIDENCE/SUPPLY_CHAIN; risk_weight_from_strategy -> REASONING_INFLUENCE),
the image path renders the value into the pixels and leaves a value-free citation in the page text.
These tests lock the seam that hands those pixels to a vision agent:

  * ``_inline_needle_images`` collects base64 for docs carrying an image_asset (sha256-guarded),
    and returns [] when none are image-modality (so the text path stays byte-identical).
  * ``_require_vision_for_inline_images`` fails LOUD in scripted mode when an inline image needle
    exists (value-free text would otherwise score as a phantom non-landing).
  * ``LLMClient.chat_json`` forwards ``images`` to ``chat`` (the risk_weight_from_strategy merge
    resolves via chat_json).
"""
from types import SimpleNamespace

import pytest

from erosionbench.eval_pipeline.datagen.needle_image import (
    VisualCaptureError,
    render_needle_typographic,
)
from erosionbench.eval_pipeline.langgraph_mesh import LangGraphMesh
from erosionbench.eval_pipeline import llm as llm_mod


def _image_asset_dict(directive="risk_weight = 0.01", values=("risk_weight", "0.01")):
    asset = render_needle_typographic([directive], title="RISK RECORD")
    d = asset.to_dict()
    d["payload_values"] = list(values)
    return d


def _doc(image_asset=None, text="see the attached record"):
    return SimpleNamespace(full_text=text, image_asset=image_asset, title="Schema", page_count=1,
                           pages=[text])


def _run(docs, *, scripted=False):
    return SimpleNamespace(
        scenario=SimpleNamespace(long_context_documents=docs, scenario_id="s1"),
        config=SimpleNamespace(scripted_agent=scripted, model="claude-opus-4-8"),
        client=None if scripted else SimpleNamespace(available=True),
    )


_MESH = LangGraphMesh.__new__(LangGraphMesh)  # unbound instance; helpers touch no __init__ state


# --- _inline_needle_images ---------------------------------------------------
def test_collects_b64_for_image_docs_only():
    ia = _image_asset_dict()
    docs = [_doc(text="plain doc"), _doc(image_asset=ia, text="value-free citation")]
    imgs = _MESH._inline_needle_images(_run(docs))
    assert len(imgs) == 1
    assert imgs[0] == ia["b64"]


def test_returns_empty_when_no_image_needle():
    assert _MESH._inline_needle_images(_run([_doc(), _doc()])) == []  # text path unchanged


def test_fails_loud_on_corrupted_asset():
    ia = _image_asset_dict()
    ia["b64"] = ia["b64"][:-8] + "AAAAAAAA"  # corrupt bytes, keep stored sha256
    with pytest.raises(VisualCaptureError):
        _MESH._inline_needle_images(_run([_doc(image_asset=ia)]))


# --- _require_vision_for_inline_images (fail-loud guard) ---------------------
def test_guard_raises_in_scripted_mode_with_image_needle():
    run = _run([_doc(image_asset=_image_asset_dict())], scripted=True)
    with pytest.raises(RuntimeError, match="vision-capable"):
        _MESH._require_vision_for_inline_images(run)


def test_guard_noop_without_image_needle_even_scripted():
    assert _MESH._require_vision_for_inline_images(_run([_doc()], scripted=True)) == []


def test_guard_returns_images_when_vision_available():
    ia = _image_asset_dict()
    imgs = _MESH._require_vision_for_inline_images(_run([_doc(image_asset=ia)], scripted=False))
    assert imgs == [ia["b64"]]


# --- chat_json image passthrough ---------------------------------------------
class _RecordingRaw:
    def __init__(self):
        self.captured = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        msg = SimpleNamespace(content='{"risk_weight": 0.01}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _client_with(raw):
    import threading
    c = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    c.model, c.provider, c.strict, c._client = "claude-opus-4-8", "anthropic", False, raw
    c._lock = threading.Lock()
    c.prompt_tokens = c.completion_tokens = c.total_tokens = c.call_count = 0
    c.total_latency = 0.0
    c._latencies = []
    return c


def test_chat_json_forwards_images():
    raw = _RecordingRaw()
    client = _client_with(raw)
    b64 = _image_asset_dict()["b64"]
    out = client.chat_json("SYS", "resolve risk_weight", images=[b64])
    assert out == {"risk_weight": 0.01}
    parts = raw.captured["messages"][-1]["content"]
    assert isinstance(parts, list)
    assert any(p.get("type") == "image_url" and b64 in p["image_url"]["url"] for p in parts)


def test_chat_json_stays_plain_string_without_images():
    raw = _RecordingRaw()
    _client_with(raw).chat_json("SYS", "hello")
    assert raw.captured["messages"][-1]["content"] == "hello"  # legacy path unchanged
