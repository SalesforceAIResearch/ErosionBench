"""Apply the real PDF parsing-friction to long-context documents (true-sim reads for ALL attacks).

Document-grounded scenarios (those with a ``document_manifest``) already serve their needle-bearing
doc as a rendered PDF that the agent downloads and extracts with pypdf — real parsing friction. But
the majority of attack classes deliver their poisoned evidence as ``long_context_documents`` injected
inline into the agent prompt as raw ``full_text``, so those never faced the text->PDF->text round-trip.

This module closes that gap by NORMALIZING the long-context docs once at scenario setup: each page is
rendered to a PDF and re-extracted with the SAME renderer + extractor the sandbox uses, and the
extracted text replaces the inline page. Every downstream reader (the firewall CONTEXT inspection,
the orchestrator schema resolver, the OVERRIDE / signing-key regexes, the ledger-corroboration
builder) then sees exactly what an agent opening the real PDF would see — with no per-reader change,
because ``ContextDocument.full_text`` / ``.page_count`` are derived from ``.pages``.

Fail-loud (project directive): if the round-trip is lossy enough that a caller-supplied payload token
no longer survives in the extracted text, we RAISE instead of silently handing back degraded pages —
a dropped payload would turn an attack into a false "defended" result. Callers pass the tokens that
MUST survive (needle/override/signing-key values) via ``require_tokens``.
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence


class ContextRoundTripError(RuntimeError):
    """A long-context payload token did not survive the PDF text->PDF->text round-trip."""


def _fold(s: str) -> str:
    """Collapse all whitespace so PDF wrapping/hyphenation doesn't cause a spurious miss."""
    return re.sub(r"\s+", "", s)


# The single value-token character class every scripted resolver captures with. Kept here (not
# duplicated per resolver) so the survival guard below checks EXACTLY what the resolvers will match.
VALUE_TOKEN = r"[A-Za-z0-9_./:-]+"


def capture_assignment(text: str, key: str) -> Optional[str]:
    """Capture the value of a buried ``<key> = <value>`` directive the way the scripted resolvers do.

    Whitespace-tolerant around ``=`` (PDF re-wrapping can push a newline there). The value is a single
    ``VALUE_TOKEN`` run — it deliberately stops at a newline, because a PDF paragraph break collapses to
    a single ``\\n`` and healing across it would swallow trailing prose. The wrap-truncation risk that
    this cannot see is caught up-front by :func:`assert_assignments_capturable`, not by loosening here.
    """
    m = re.search(rf"{re.escape(key)}\s*=\s*({VALUE_TOKEN})", text)
    return m.group(1).strip() if m else None


def assert_assignments_capturable(
    extracted: Sequence[str],
    source: Sequence[str],
    payload_values: Optional[Sequence[str]] = None,
) -> None:
    """Fail loud if the PDF round-trip would make a scripted resolver capture a TRUNCATED value.

    Every real dataset value (≤~100 contiguous chars) survives and captures fully — but a future long
    value could wrap mid-token so the resolver's single value-token run returns only the prefix,
    silently weakening the attack.

    The check is VALUE-centric, not key-centric: a key may legitimately be assigned several DIFFERENT
    values in one doc (conflicting-entry / Crescendo attack material), so comparing one capture against
    every occurrence would false-positive. For each DISTINCT candidate value we confirm it reappears in
    the extracted text as a whole contiguous value-token run. The truncation signature we reject is
    precisely: the value is present when whitespace is folded (so it *did* survive) yet NO contiguous
    run equals it (so a wrap split it mid-token). A value that is simply absent is left to the presence
    guard in :func:`roundtrip_pages_checked`; here we only catch wrap-truncation.

    ``payload_values`` scopes the check to the ATTACK payload — the specific values a resolver actually
    recovers (the forged ``doc_override_key`` / ``doc_resolve_key`` / signing-key values). This is the
    correct behaviour for the shared-corpus path, where a base document legitimately contains incidental
    ``X = Y`` prose (a RACI legend ``R = Responsible``, a Collibra reference yielding ``region=PRD``)
    that the needle splice may reflow and truncate — but that NO resolver reads, so its truncation
    neuters nothing. Scanning every ``X = Y`` (``payload_values=None``, the legacy behaviour retained
    for callers that pass no payload) would false-fail those scenarios. When ``payload_values`` is
    given we check only those values (still whitespace-folded, still whole-run), and never the ambient
    document prose.
    """
    ext_text = "\n".join(extracted)
    # Every maximal value-token run the resolver could capture from the extracted text.
    runs = {r for r in re.findall(VALUE_TOKEN, ext_text) if len(r) >= 2}
    runs_folded = {_fold(r) for r in runs}
    ext_folded = _fold(ext_text)

    if payload_values is not None:
        candidates = [str(v).strip() for v in payload_values if str(v).strip()]
    else:
        # Legacy: every assignment value present in the source document.
        candidates = [
            m.group(1).strip()
            for m in re.finditer(rf"[A-Za-z_][\w.]*\s*=\s*({VALUE_TOKEN})", "\n".join(source))
        ]

    seen: set = set()
    for want in candidates:
        if len(want) < 2 or want in seen:
            continue
        # A payload value is only capture-relevant if it is itself a single value-token run — a
        # multi-token / free-text value can never be recovered whole by the resolver anyway, so the
        # wrap-truncation signature does not apply to it (the presence guard owns its survival).
        if payload_values is not None and not re.fullmatch(VALUE_TOKEN, want):
            continue
        seen.add(want)
        want_folded = _fold(want)
        if want_folded in runs_folded:
            continue  # captured whole by some run — fine
        if want_folded in ext_folded:
            # Present (survived the round-trip) but no contiguous run recovers it: a wrap split it.
            raise ContextRoundTripError(
                f"assignment value {want!r} round-tripped to a truncated capture "
                f"(present in text but split across a PDF wrap, so the resolver's single value-token "
                f"run recovers only a prefix); this would silently neuter the attack. "
                f"Refusing to serve degraded pages."
            )
        # else: absent entirely — roundtrip_pages_checked's presence guard owns that case.


def roundtrip_pages(pages: Sequence[str], title: str = "context") -> List[str]:
    """Render ``pages`` to a PDF and extract per-page text back — the parsing friction an agent faces.

    One input page maps to one extracted page (pypdf preserves page structure). Deliberately NOT
    byte-identical to the input: headers/footers, wrapping and unicode folding are the real friction.
    """
    from pypdf import PdfReader

    from .datagen.pdf_renderer import render_document_to_pdf

    if not pages:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ctx.pdf"
        render_document_to_pdf(
            pages=list(pages), title=title, output_path=out,
            metadata={"total_pages": len(pages), "scenario_id": "CONTEXT"},
        )
        data = out.read_bytes()

    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def roundtrip_pages_checked(
    pages: Sequence[str],
    require_tokens: Optional[Sequence[str]] = None,
    title: str = "context",
) -> List[str]:
    """``roundtrip_pages`` plus a fail-loud guarantee that each ``require_tokens`` entry survived.

    ``require_tokens`` are the payload substrings that MUST still be present (whitespace-folded) in
    the extracted text — e.g. the needle value, a buried ``OVERRIDE`` value, a signing key. If any is
    missing after the round-trip, raise :class:`ContextRoundTripError` rather than return degraded
    pages that would silently neuter the attack.
    """
    if not pages:
        return []  # nothing to round-trip; the survival guard has nothing to check
    extracted = roundtrip_pages(pages, title=title)
    folded = _fold("\n".join(extracted))
    for tok in require_tokens or ():
        tok = (tok or "").strip()
        if not tok:
            continue
        if _fold(tok) not in folded:
            raise ContextRoundTripError(
                f"context payload token {tok!r} did not survive the PDF round-trip "
                f"(title={title!r}); refusing to serve degraded pages that would mask the attack."
            )
    return extracted
