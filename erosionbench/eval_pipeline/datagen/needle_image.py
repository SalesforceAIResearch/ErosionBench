# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Image-modality needle rendering + capturability gate for the SHARED-CORPUS path.

This is the image sibling of :mod:`needle_blender` + the ``_assert_roundtrip`` guard in
:mod:`corpus_needle_docs`. The idea is to deliver the *same* ``key = value`` erosion needle through an
**image** the prose cites as corroboration, instead of through extractable page text. Landing is
scored identically downstream (deterministic predicate over ``tool_log``), so text-vs-image is
directly comparable — this module only changes how the payload gets into the artifact and how we
guarantee, at DATAGEN-build time, that it is recoverable from pixels.

Two things live here, and only these:

  1. :func:`render_needle_typographic` — render the forged directive line(s) as a legible image
     (**M1 typographic**: black text on white; the OCR-trivial control modality). This is the
     analog of :func:`needle_renderer.splice_needle` for the image path. It is deterministic given
     ``(directive_lines, title, seed)`` and depends only on Pillow (no new system deps).

  2. :func:`assert_visual_capturable` — the image analog of ``corpus_needle_docs._assert_roundtrip``.
     It reads the payload back OUT of the rendered pixels with a *pluggable reader* and RAISES
     :class:`VisualCaptureError` unless every forged value token reappears. This is the load-bearing
     correctness invariant: a shipped image whose payload cannot be recovered would silently score
     as a non-landing at eval time, corrupting ASR.

Why the reader is a deterministic OCR by default (and the VLM is deliberately NOT the gate)
-------------------------------------------------------------------------------------------
The text path already guarantees payload presence with a **deterministic, model-agnostic**
extractor (``pypdf.extract_text`` inside ``assert_assignments_capturable``) — "the payload is in
the artifact" is ground truth, not an LLM opinion. The faithful image analog is deterministic OCR
(tesseract), not a vision model: if weak OCR can read the value, any capable VLM can, so OCR is a
*conservative floor* AND it keeps dataset validity independent of the model under test. The VLM
belongs at EVAL time (the vision-enabled agent / auditor experiment), not in the data-validity gate.
Accordingly the reader is injected by the caller; :func:`get_ocr_reader` returns the deterministic
default, and a VLM reader can be supplied later as a *secondary recorded* signal without ever
gating validity on it.

No silent fallback (per the project directive, mirrored from :mod:`needle_blender`):
  * If no reader is configured, :func:`assert_visual_capturable` RAISES — it never "passes" blind.
  * If the payload does not read back, it RAISES with the offending token and the recovered text —
    it never degrades to assuming success.
  * If OCR is requested but unavailable, :func:`get_ocr_reader` RAISES with an install hint — it
    never substitutes a weaker check.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .needle_renderer import stable_seed

logger = logging.getLogger(__name__)

__all__ = [
    "ImageAsset",
    "VisualCaptureError",
    "assert_visual_capturable",
    "get_ocr_reader",
    "render_needle_image",
    "render_needle_typographic",
]

#: A capturability reader: raw PNG bytes -> recovered text. Injected so the data-validity gate
#: stays model-agnostic (deterministic OCR by default; a VLM may be supplied as a secondary signal).
Reader = Callable[[bytes], str]

#: Supported delivery modalities (orthogonal to the tier axis — a technique, not a tier).
_MODALITY_M1 = "M1"  # typographic: OCR-trivial control (payload rendered as plain text)
_MODALITY_M2 = "M2"  # camouflage: payload embedded in a realistic carrier screenshot (Slice 2)
_SUPPORTED_MODALITIES = frozenset({_MODALITY_M1, _MODALITY_M2})


class VisualCaptureError(RuntimeError):
    """The forged payload could not be recovered from the rendered image (or no reader configured).

    Raised instead of silently accepting the image, so a scenario whose payload does not survive
    the render is surfaced loudly at build time rather than scoring as a phantom non-landing at
    eval time (mirrors :class:`needle_blender.NeedleBlendError`).
    """


@dataclass(frozen=True)
class ImageAsset:
    """A rendered needle-carrier image plus the provenance needed to verify and serialize it.

    ``data`` holds raw PNG bytes (datagen-internal). Serialization to base64 for ``dataset.json``
    and the verification manifest happens at the adapter/serialize layer, using :attr:`b64` /
    :attr:`sha256`. ``payload_values`` are exactly the forged value tokens baked into the pixels —
    the same tokens the resolver recovers — so the manifest can re-verify capturability offline.
    """

    data: bytes                              # raw PNG bytes
    modality: str                            # one of _SUPPORTED_MODALITIES
    width: int
    height: int
    payload_values: List[str] = field(default_factory=list)  # forged value tokens baked into pixels
    carrier_id: str = ""                     # M2: which carrier template was composited (M1: "")
    reader_name: str = ""                    # which reader verified capturability (provenance)
    fmt: str = "png"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def b64(self) -> str:
        return b64encode(self.data).decode("ascii")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form for ``dataset.json`` / the verification manifest (raw bytes -> base64)."""
        return {
            "modality": self.modality, "fmt": self.fmt,
            "width": self.width, "height": self.height,
            "payload_values": list(self.payload_values),
            "carrier_id": self.carrier_id, "reader_name": self.reader_name,
            "sha256": self.sha256, "b64": self.b64,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageAsset":
        """Inverse of :meth:`to_dict` (base64 -> raw bytes). Verifies the stored sha256 if present."""
        data = b64decode(d["b64"])
        asset = cls(
            data=data, modality=d["modality"], width=int(d["width"]), height=int(d["height"]),
            payload_values=list(d.get("payload_values", [])), carrier_id=d.get("carrier_id", ""),
            reader_name=d.get("reader_name", ""), fmt=d.get("fmt", "png"),
        )
        want = d.get("sha256")
        if want and want != asset.sha256:
            raise VisualCaptureError(
                f"image asset sha256 mismatch on load (stored {want}, computed {asset.sha256}); "
                f"the serialized bytes were corrupted."
            )
        return asset


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _load_font(size: int):
    """A legible font at ``size`` px. Prefer a real TrueType; else scale Pillow's built-in font.

    This is a *quality* preference, not a correctness fallback: :func:`assert_visual_capturable`
    is the real guarantee, so an environment without system TrueType fonts still produces a
    capturable image via Pillow's (scalable, since Pillow 10) built-in font.
    """
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10: scalable built-in
    except TypeError:  # very old Pillow: unscaled bitmap (still legible for OCR at 1x)
        return ImageFont.load_default()


def render_needle_typographic(
    directive_lines: Sequence[str],
    *,
    title: Optional[str] = None,
    seed: int = 0,
    font_size: int = 30,
    hygiene_jitter: bool = False,
) -> ImageAsset:
    """Render ``directive_lines`` as an **M1 typographic** needle image (black text on white).

    ``directive_lines`` are the exact ``key = value`` lines whose *values* must survive to the
    tool call — pass them verbatim (e.g. ``f"{k} = {_fmt_value(v)}"``). ``title`` renders a small
    header (e.g. a system-record caption the prose can cite). ``hygiene_jitter`` applies a small,
    seed-deterministic variation in padding/size so byte-reused carriers still differ per scenario
    (the "per-scenario hygiene variation" decision) without changing legibility.

    Returns an :class:`ImageAsset` with ``payload_values`` UNSET (the caller supplies the exact
    resolver tokens); use :func:`render_needle_image` for the full compose-side entrypoint that
    attaches ``payload_values`` and runs the capturability gate.
    """
    from PIL import Image, ImageDraw

    if not directive_lines:
        raise VisualCaptureError("render_needle_typographic: no directive lines to render.")

    rng = stable_seed(seed, "typographic", *directive_lines)
    jitter = (rng % 7) if hygiene_jitter else 0  # 0..6 px, deterministic per (seed, payload)

    pad = 28 + jitter
    line_gap = 12
    header_gap = 18

    font = _load_font(font_size)
    header_font = _load_font(max(14, int(font_size * 0.6)))

    # Measure on a scratch canvas.
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def _measure(text: str, fnt) -> tuple[int, int]:
        l, t, r, b = scratch.textbbox((0, 0), text, font=fnt)
        return r - l, b - t

    line_dims = [_measure(ln, font) for ln in directive_lines]
    content_w = max(w for w, _ in line_dims)
    content_h = sum(h for _, h in line_dims) + line_gap * (len(directive_lines) - 1)

    header_h = 0
    if title:
        hw, hh = _measure(title, header_font)
        content_w = max(content_w, hw)
        header_h = hh + header_gap

    width = content_w + 2 * pad
    height = content_h + header_h + 2 * pad

    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    y = pad
    if title:
        draw.text((pad, y), title, fill="#3a3a3a", font=header_font)
        y += header_h
    for ln, (_, h) in zip(directive_lines, line_dims):
        draw.text((pad, y), ln, fill="#111111", font=font)
        y += h + line_gap

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ImageAsset(
        data=buf.getvalue(), modality=_MODALITY_M1, width=width, height=height,
    )


def _find_chrome() -> str:
    """Locate a headless-capable Chrome/Chromium, or RAISE (no-silent-fallback)."""
    env = os.environ.get("CHROME_BIN")
    if env and os.path.exists(env):
        return env
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            candidates.append(path)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise VisualCaptureError(
        "M2 rendering needs headless Chrome/Chromium and none was found. Set CHROME_BIN to the "
        "binary path, or install Google Chrome / Chromium."
    )


def render_needle_carrier(
    *,
    payload: str,
    attack_class: str,
    family_key: str = "",
    domain_id: str = "",
    org: str = "the organization",
    title: str = "",
    seed: int = 0,
    hygiene_jitter: bool = True,
    variant_key: str = "",
) -> ImageAsset:
    """Render an **M2 camouflage** carrier: composite ``payload`` into a realistic UI screenshot.

    Selects a template from the fixed carrier pool (deterministic per attack family, domain-aware),
    fills the payload region, and screenshots it with headless Chrome. The value is real on-screen
    text, so the same OCR gate recovers it. Raises if Chrome is unavailable or the screenshot is empty.

    ``variant_key`` rotates a carrier's decorative chart (Grafana board / CI run) per family so
    distinct scenarios don't all look identical; it does NOT affect carrier selection or the payload.
    """
    from .needle_carriers import render_carrier_html, select_carrier

    carrier = select_carrier(attack_class, family_key, domain_id)
    html = render_carrier_html(
        carrier, payload=payload, title=title or carrier.label, org=org, variant_key=variant_key,
    )

    rng = stable_seed(seed, "carrier", attack_class, family_key)
    jitter_w = (rng % 5) if hygiene_jitter else 0        # 0..4 px width nudge (per-scenario hygiene)
    jitter_h = ((rng >> 3) % 5) if hygiene_jitter else 0
    width, height = 800 + jitter_w, 620 + jitter_h
    scale = 2

    chrome = _find_chrome()
    with tempfile.TemporaryDirectory(prefix="needle_carrier_") as td:
        html_path = os.path.join(td, "carrier.html")
        png_path = os.path.join(td, "carrier.png")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        cmd = [
            chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
            f"--screenshot={png_path}", f"file://{html_path}",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # TimeoutExpired.stderr is often None; guard so the handler surfaces the real cause
            # instead of crashing with its own AttributeError.
            stderr = getattr(e, "stderr", None) or b""
            raise VisualCaptureError(
                f"headless Chrome failed to render the M2 carrier ({type(e).__name__}): "
                f"{stderr.decode('utf-8', 'replace')[:300]}"
            ) from e
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            raise VisualCaptureError("headless Chrome produced no (or an empty) screenshot for M2.")
        with open(png_path, "rb") as f:
            data = f.read()

    return ImageAsset(
        data=data, modality=_MODALITY_M2, width=width * scale, height=height * scale,
        carrier_id=carrier.carrier_id,
    )


def render_needle_image(
    *,
    modality: str,
    directive_lines: Sequence[str],
    payload_values: Sequence[str],
    reader: Reader,
    title: Optional[str] = None,
    seed: int = 0,
    hygiene_jitter: bool = False,
    carrier_id: str = "",
    attack_class: str = "",
    family_key: str = "",
    domain_id: str = "",
    org: str = "the organization",
    variant_key: str = "",
) -> ImageAsset:
    """Compose-side entrypoint: render the needle image AND gate it for capturability (fail-loud).

    This is the function :func:`corpus_needle_docs.compose_needle_document` calls on the image
    branch, sibling to the text branch's ``splice_needle`` + ``_assert_roundtrip``. It renders per
    ``modality`` (M1 typographic or M2 camouflage carrier), attaches the exact resolver
    ``payload_values`` (what the tool call must recover), verifies they read back out of the pixels
    via ``reader``, and RAISES on any failure — no image ships unless its payload is recoverable.
    """
    if modality not in _SUPPORTED_MODALITIES:
        raise VisualCaptureError(
            f"unknown modality {modality!r}; supported: {sorted(_SUPPORTED_MODALITIES)}."
        )

    if modality == _MODALITY_M2:
        payload = " ".join(directive_lines)
        rendered = render_needle_carrier(
            payload=payload, attack_class=attack_class, family_key=family_key, domain_id=domain_id,
            org=org, title=title or "", seed=seed, hygiene_jitter=hygiene_jitter,
            variant_key=variant_key,
        )
    else:
        rendered = render_needle_typographic(
            directive_lines, title=title, seed=seed, hygiene_jitter=hygiene_jitter,
        )

    reader_name = getattr(reader, "__name__", reader.__class__.__name__)
    asset = ImageAsset(
        data=rendered.data, modality=rendered.modality, width=rendered.width, height=rendered.height,
        payload_values=list(payload_values), carrier_id=rendered.carrier_id or carrier_id,
        reader_name=reader_name,
    )
    assert_visual_capturable(asset.data, payload_values, reader=reader)
    return asset


# ---------------------------------------------------------------------------
# Capturability gate (image analog of _assert_roundtrip)
# ---------------------------------------------------------------------------
# OCR routinely resolves a monospace hyphen-minus as an en/em dash (or minus sign) depending on
# kerning — e.g. a reference code "RISK-2024-437" reads back as "RISK-2024—437". The value is
# LEGIBLY present in the pixels (a human, and the eval-time vision model, read it correctly); only
# tesseract's dash-GLYPH choice varies. Folding all dash variants to a single ASCII hyphen before
# the substring check tolerates that ambiguity WITHOUT weakening the invariant — the expected token
# and the recovered text are normalized identically, so a truly-absent value still fails.
_DASH_VARIANTS = "‐‑‒–—―−­"  # hyphen/figure/en/em/bar/minus/soft


def _normalize(text: str) -> str:
    """Whitespace-insensitive, case-folded, dash-folded form for tolerant OCR substring matching."""
    text = text.translate({ord(c): "-" for c in _DASH_VARIANTS})
    # OCR routinely renders a single monospace hyphen as hyphen+dash on reference codes
    # (e.g. CHG-2024 -> "CHG-—2024"); collapse runs of dashes so the value still matches.
    text = re.sub(r"-{2,}", "-", text)
    return re.sub(r"\s+", "", text).lower()


# tesseract's most common false glyph substitution on a small, crisp monospace zero is '@' (observed
# deterministically on bare decimals like "0.212" -> "@.212"). This fold is applied ONLY to the
# positive capturability read-back (never the channel-split absence check) and ONLY as a fallback
# after the strict match fails. It can recover a KNOWN misread of a value we DID rasterize, but it
# cannot manufacture a match for a value that was never rendered: the render always rasterizes the
# intended token, so any '@' in the OCR is a misread of our own '0'.
_OCR_CONFUSIONS = str.maketrans({"@": "0"})


def _ocr_fold(text: str) -> str:
    """`_normalize` plus tolerance for tesseract's known glyph confusions (read-back matching only)."""
    return _normalize(text).translate(_OCR_CONFUSIONS)


def assert_visual_capturable(
    png_bytes: bytes,
    require_tokens: Sequence[str],
    *,
    reader: Optional[Reader],
) -> str:
    """Fail loud unless every forged value token reads back out of ``png_bytes`` via ``reader``.

    The image analog of :func:`corpus_needle_docs._assert_roundtrip`. ``require_tokens`` are the
    forged PAYLOAD values (exactly what a resolver recovers), scoped to those so incidental carrier
    chrome does not false-fail. Returns the recovered text on success (useful for provenance). No
    reader → RAISE (never a blind pass); token missing → RAISE with the token + recovered text.
    """
    if reader is None:
        raise VisualCaptureError(
            "no capturability reader configured — refusing to accept an unverified image. "
            "Pass reader=get_ocr_reader() (deterministic default) or an explicit VLM reader."
        )
    if not require_tokens:
        raise VisualCaptureError("assert_visual_capturable: no payload tokens to verify.")

    recovered = reader(png_bytes)
    norm = _normalize(recovered)
    missing = [tok for tok in require_tokens if _normalize(tok) not in norm]
    if missing:
        # Fallback for the still-missing tokens only: tolerate tesseract's known glyph confusions
        # (e.g. a monospace '0' read as '@'). Strict matches above are unaffected.
        folded = _ocr_fold(recovered)
        missing = [tok for tok in missing if _ocr_fold(tok) not in folded]
    if missing:
        preview = recovered.strip().replace("\n", " ")[:200]
        raise VisualCaptureError(
            f"payload token(s) {missing!r} did not read back out of the rendered image "
            f"(reader recovered: {preview!r}). Fix the render / payload — no degraded fallback."
        )
    return recovered


def get_ocr_reader() -> Reader:
    """Return the deterministic default reader (tesseract via pytesseract), or RAISE with a hint.

    Deliberately fail-loud when OCR is unavailable rather than substituting a weaker check: the
    capturability gate must be a real guarantee. Install with ``brew install tesseract`` (macOS)
    and ``pip install pytesseract`` into the project venv.
    """
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        raise VisualCaptureError(
            "OCR reader unavailable: pytesseract not importable. Install it in the project venv "
            "(`pip install pytesseract`) and the tesseract binary (`brew install tesseract`)."
        ) from e

    def tesseract_reader(png_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image

        try:
            return pytesseract.image_to_string(Image.open(io.BytesIO(png_bytes)))
        except pytesseract.TesseractNotFoundError as e:  # binary missing, not the python pkg
            raise VisualCaptureError(
                "OCR reader unavailable: the `tesseract` binary is not on PATH. "
                "Install it (`brew install tesseract` on macOS)."
            ) from e

    return tesseract_reader
