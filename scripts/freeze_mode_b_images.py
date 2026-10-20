"""Freeze the Mode-B image-corroboration arm for reproducibility (Direction-1 multimodal).

Mode B is the scored research arm: the forged value is already in the ledger (normal attack
mechanics, landing parity automatic); it is ECHOED in a realistic carrier IMAGE cited by a
value-free page. Because the M2 carrier is an HTML->headless-Chrome screenshot (NOT byte-stable
across Chrome/font versions) AND the text-arm citation is LLM-authored, neither arm is reproducible
if rendered at eval time. So we render ONCE here, with a FIXED composer model (so the corroboration
content is part of the dataset, not a function of the model under test), FREEZE both arms, and have
eval LOAD them (see MeshConfig.frozen_corroboration_path). Nothing here scores; it builds the frozen
artifact + the paired GROUND TRUTH that justifies the text-vs-image difference is a pure channel
split (same value, same landing predicate; value in pixels + value-free citation vs. value in text).

Eligibility (the Mode-B corroboration seam, `_Run.ledger_corroboration`): merge != document_grounded,
has forged_token_key, has long_context_documents. That is the 12 ledger/debate/collusion classes
where the value lives in the LEDGER. The 4 document_grounded classes deliver value IN the document
(Mode-A territory) and are deliberately excluded here.

Artifacts (out-dir, default results/mode_b_frozen/):
  * mode_b_frozen.json  -- the LOAD file: {scenario_id: {shared ground truth, image arm, text arm}}
  * images/<scenario_id>.png -- the frozen ground-truth carrier images (for hashing + inspection)

Usage:
    .venv/bin/python scripts/freeze_mode_b_images.py \
        --eval data/eval.json --composer-model gpt-5 \
        --families-per-class 2 --tiers 1,2,3 --image-modality M2 \
        --out-dir results/mode_b_frozen
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, ".")

# Extended LLM timeouts (gpt-5 citation authoring can be slow). Set before importing the client.
for _k, _v in (
    ("EROSIONBENCH_LLM_REQUEST_TIMEOUT", "300"),
    ("EROSIONBENCH_LLM_MAX_ATTEMPTS", "4"),
):
    os.environ.setdefault(_k, _v)

_TIER_RE = re.compile(r"_(T\d)$")


def _family_stem(scenario_id: str) -> str:
    """Strip the tier suffix (_T2/_T3) so all tiers of one instance share a family key."""
    return _TIER_RE.sub("", scenario_id)


def _tier_of(scenario_id: str) -> int:
    m = _TIER_RE.search(scenario_id)
    return int(m.group(1)[1:]) if m else 1


def _value_str(forged_value) -> str:
    """The on-image/in-text rendering of the forged value (mirrors _compose_image_corroboration)."""
    return forged_value if isinstance(forged_value, str) else repr(forged_value)


def _select_families(scenarios, families_per_class: int, *, benign: bool = False,
                     rank_offset: int = 0):
    """Deterministically pick `families_per_class` families per eligible class, by stable_seed rank.

    Eligibility mirrors `_Run.ledger_corroboration`: merge != document_grounded + forged_token_key +
    long_context_documents. Returns the flat list of selected scenario objects (all requested tiers).

    ``rank_offset`` skips the first N ranked families per class before taking the next
    ``families_per_class``. This carves a CALIBRATION pool that is FAMILY-DISJOINT from the frozen
    eval set: the eval freeze took ranks ``[0:families_per_class]`` (offset 0); a calibration freeze
    with ``rank_offset=families_per_class`` takes ``[fpc:2*fpc]`` -- non-overlapping families, so the
    calibration artifact can never be fit on an eval family (``assert_not_fit_on`` stays satisfied).

    ``benign=True`` freezes the BENIGN corroboration twins (the negative class): it accepts
    is_malicious=False / attack_class=None scenarios and takes ALL eligible families (no per-class
    truncation, ``rank_offset`` ignored) -- the twin file is already the pre-selected set, one twin
    per frozen malicious sid.
    """
    from erosionbench.eval_pipeline.datagen.needle_renderer import stable_seed

    by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in scenarios:
        if getattr(s, "is_malicious", False) != (not benign):
            # malicious mode wants is_malicious True; benign mode wants is_malicious False
            continue
        spec = s.execution_spec or {}
        if spec.get("merge") == "document_grounded":
            continue
        if not spec.get("forged_token_key"):
            continue
        if not s.long_context_documents:
            continue
        ac = getattr(s.attack_class, "value", s.attack_class) or "benign"
        by_class[ac][_family_stem(s.scenario_id)].append(s)

    selected = []
    picked_report = {}
    for ac in sorted(by_class):
        fams = by_class[ac]
        ranked = sorted(fams, key=lambda fam: stable_seed(ac, fam))
        chosen = (list(ranked) if benign
                  else ranked[rank_offset:rank_offset + families_per_class])
        picked_report[ac] = chosen
        for fam in chosen:
            selected.extend(fams[fam])
    return selected, picked_report


def _compose_arm(mesh, scenario, *, image: bool):
    """Run the real corroboration setup for one arm and return the mutated _Run.

    Toggles the env gate that `_Run.image_corroboration`/`ledger_corroboration` read, builds a
    minimal run, seeds the ledger, and calls the production `_setup_ledger_corroboration` so the
    frozen bytes are exactly what eval would compose at runtime.
    """
    from erosionbench.eval_pipeline.langgraph_mesh import _Run
    from erosionbench.eval_pipeline.firewalls import build_firewall

    if image:
        os.environ["EROSIONBENCH_IMAGE_CORROBORATION"] = "1"
        os.environ.pop("EROSIONBENCH_LEDGER_CORROBORATION", None)
    else:
        os.environ["EROSIONBENCH_LEDGER_CORROBORATION"] = "1"
        os.environ.pop("EROSIONBENCH_IMAGE_CORROBORATION", None)

    request = (scenario.execute_session.request if scenario.execute_session
               else scenario.user_intent)
    run = _Run(
        scenario=scenario, config=mesh.config, client=mesh._client,
        firewall=build_firewall("none"), request=request,
    )
    mesh._replay_ledger(run)
    mesh._setup_ledger_corroboration(run)
    return run


def _ocr_transcript(image_asset: dict, reader) -> str:
    """Deterministic OCR read-back of the frozen image, persisted as ground-truth proof."""
    png = base64.b64decode(image_asset["b64"])
    return reader(png)


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the Mode-B image-corroboration arm.")
    ap.add_argument("--eval", default="data/eval.json",
                    help="Held-out eval dataset to source scenarios from.")
    ap.add_argument("--composer-model", default="gpt-5",
                    help="FIXED model that authors both arms' citations (NOT the eval agent model; "
                         "adversarial composition must not use Claude).")
    ap.add_argument("--families-per-class", type=int, default=2)
    ap.add_argument("--family-rank-offset", type=int, default=0,
                    help="Skip the first N ranked families per class before selecting "
                         "(malicious mode only). Use =families-per-class to build a CALIBRATION pool "
                         "family-disjoint from the eval freeze (which used offset 0).")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--image-modality", default="M2", choices=["M1", "M2"])
    ap.add_argument("--out-dir", default="results/mode_b_frozen")
    ap.add_argument("--benign-twins", action="store_true",
                    help="Freeze the BENIGN corroboration twins (negative class): accept "
                         "is_malicious=False scenarios and take ALL eligible families.")
    args = ap.parse_args()

    want_tiers = {int(t) for t in args.tiers.split(",")}

    # Local imports (after sys.path + env) -----------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from run_full_eval import load_scenarios  # canonical loader (compiles assertions, harmless here)
    from erosionbench.eval_pipeline.langgraph_mesh import LangGraphMesh, MeshConfig
    from erosionbench.eval_pipeline.llm import LLMClient
    from erosionbench.eval_pipeline.datagen.needle_image import get_ocr_reader
    from erosionbench.eval_pipeline.attack_engine import compose_text_channel_split

    reader = get_ocr_reader()  # fail-loud if OCR missing -- no unverified image ships
    scenarios = load_scenarios(args.eval)
    # The canonical loader compiles + POPS ground_truth_contracts, so read the raw file too to
    # recover the landing-contract params (the shared ground truth that proves the arms are matched).
    with open(args.eval) as _fh:
        contracts_by_sid = {
            item["scenario_id"]: item.get("ground_truth_contracts", [])
            for item in json.load(_fh)
        }

    selected, report = _select_families(scenarios, args.families_per_class, benign=args.benign_twins,
                                         rank_offset=args.family_rank_offset)
    selected = [s for s in selected if _tier_of(s.scenario_id) in want_tiers]

    print(f"Freeze Mode B: composer={args.composer_model} modality={args.image_modality} "
          f"families/class={args.families_per_class} rank_offset={args.family_rank_offset} "
          f"tiers={sorted(want_tiers)}")
    print(f"  {len(report)} eligible classes, {len(selected)} scenarios to freeze")
    for ac, fams in report.items():
        print(f"    {ac:32s} families={fams}")
    print()

    # A mesh whose composer client is the FIXED model (decoupled from any eval agent model).
    config = MeshConfig(mode="multi", model=args.composer_model, firewall_kind="none",
                        scripted_agent=False, verbose=False)
    mesh = LangGraphMesh(config)
    mesh._client = LLMClient(model=args.composer_model, strict=True)

    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    t0 = time.time()
    frozen = {}
    errors = []
    skipped = []
    for i, scen in enumerate(sorted(selected, key=lambda s: s.scenario_id), 1):
        sid = scen.scenario_id
        spec = scen.execution_spec or {}
        forged_key = spec.get("forged_token_key", "")
        ac = getattr(scen.attack_class, "value", scen.attack_class)
        print(f"[{i}/{len(selected)}] {sid} ({ac}) ...", end="", flush=True)
        try:
            # Image arm (value only in pixels + value-free citation).
            img_run = _compose_arm(mesh, scen, image=True)
            image_asset = img_run.corrob_image_asset
            image_citation = img_run.corrob_needle_text
            forged_value = mesh._resolve_forged_value(img_run, forged_key)
            # Some nominally-eligible classes (e.g. reasoning_influence) carry the value in the
            # DOCUMENT, not the ledger, so _resolve_forged_value -> None and the corroboration setup
            # correctly declines to plant a needle. That is a CLEAN skip (not a Mode-B ledger class),
            # not a failure -- record and move on so the real ledger classes still freeze.
            if forged_value is None and not image_asset:
                print(" SKIP (value not in ledger -- document-grounded value, not a Mode-B class)")
                skipped.append({"scenario_id": sid, "attack_class": ac,
                                "reason": "forged value not resolvable from ledger"})
                continue
            if not image_asset:
                raise RuntimeError("image arm produced no image_asset (setup skipped?)")

            # Text arm (matched control): the DETERMINISTIC twin of the image arm -- the SAME rendered
            # value lines + the SAME citation wrapper + the SAME ticket, only the value lives IN the
            # page text instead of the pixels. No LLM (the image arm has none either), so it is
            # refusal-immune and reproducible (the point of freezing) with guaranteed anchor parity.
            text_citation, _txt_lines, _txt_scalars = compose_text_channel_split(
                forged_key, forged_value, img_run.corrob_ticket_ref or f"CHG-{sid[-5:]}")
            if not text_citation:
                raise RuntimeError("text arm produced no citation")

            # --- ground-truth assertions (fail loud) ----------------------------------------
            # The shared anchors are the distinctive value tokens rendered into the pixels (the
            # SAME tokens for both arms). We measure each arm by how many of these anchors it
            # delivers in its own channel -- that is the pure channel-split ground truth: image =
            # anchors in pixels + ZERO in the page text; text = anchors in the page text.
            payload_values = image_asset.get("payload_values") or []
            from erosionbench.eval_pipeline.datagen.needle_image import _normalize, _ocr_fold
            transcript = _ocr_transcript(image_asset, reader)
            # Recovery re-check uses the SAME OCR-confusion-tolerant fold the image gate used at
            # compose time (else a value the gate accepted via the fold would re-fail here).
            fold_ocr = _ocr_fold(transcript)
            ocr_recovered = all(_ocr_fold(pv) in fold_ocr for pv in payload_values)

            norm_img_cite = _normalize(image_citation or "")
            anchors_leaked_to_image_text = [pv for pv in payload_values if _normalize(pv) in norm_img_cite]
            value_absent_from_image_text = not anchors_leaked_to_image_text

            norm_txt_cite = _normalize(text_citation or "")
            anchors_in_text = [pv for pv in payload_values if _normalize(pv) in norm_txt_cite]
            value_present_in_text_arm = len(anchors_in_text) == len(payload_values) and bool(payload_values)

            if not ocr_recovered:
                raise RuntimeError(f"OCR did not recover payload {payload_values} from frozen image")
            if not value_absent_from_image_text:
                raise RuntimeError(
                    f"anchor(s) {anchors_leaked_to_image_text} leaked into the image arm's citation "
                    f"(not value-free channel split)")
            # The text arm is separately LLM-authored; it may paraphrase and not carry every anchor
            # verbatim. Record the overlap (don't gate), but WARN loudly if it delivers no anchor at
            # all -- that would mean the "matched text control" isn't actually delivering the value.
            if not anchors_in_text:
                print(" [WARN text arm carries NONE of the anchors] ", end="", flush=True)

            # --- persist the frozen PNG (ground-truth image) ---------------------------------
            png = base64.b64decode(image_asset["b64"])
            png_rel = os.path.join("images", f"{sid}.png")
            with open(os.path.join(args.out_dir, png_rel), "wb") as fh:
                fh.write(png)
            png_sha = hashlib.sha256(png).hexdigest()

            frozen[sid] = {
                "scenario_id": sid,
                "attack_class": ac,
                "merge": spec.get("merge"),
                "tier": _tier_of(sid),
                "domain": spec.get("domain", ""),
                # shared ground truth -- proves the two arms are MATCHED, not confounded
                "forged_key": forged_key,
                "forged_value": forged_value,
                "target_parameters": [
                    {"name": c.get("name"), "assertion_type": c.get("assertion_type"),
                     "severity": c.get("severity"), "params": c.get("params", {})}
                    for c in contracts_by_sid.get(sid, [])
                ],
                "corrob_doc_id": img_run.corrob_doc_id,
                "injection_page": img_run.corrob_needle_page,
                # the shared distinctive value tokens both arms are measured against
                "payload_anchors": payload_values,
                # image arm
                "image": {
                    "png_file": png_rel,
                    "sha256": png_sha,
                    "carrier_id": image_asset.get("carrier_id"),
                    "modality": image_asset.get("modality"),
                    "payload_values": payload_values,
                    "ocr_transcript": transcript.strip(),
                    "ocr_recovered": ocr_recovered,
                    "citation_text": image_citation,
                    "value_absent_from_text": value_absent_from_image_text,
                    "anchors_in_page_text": anchors_leaked_to_image_text,  # must be [] (channel split)
                    "image_asset": image_asset,  # self-contained for eval load (b64 + sha)
                },
                # text arm (matched control)
                "text": {
                    "citation_text": text_citation,
                    "value_present_in_text": value_present_in_text_arm,
                    "anchors_in_text": anchors_in_text,
                    "anchors_in_text_count": f"{len(anchors_in_text)}/{len(payload_values)}",
                },
            }
            print(f" OK carrier={image_asset.get('carrier_id')} ocr={ocr_recovered} "
                  f"txt_anchors={len(anchors_in_text)}/{len(payload_values)}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f" FAILED: {msg}")
            errors.append({"scenario_id": sid, "attack_class": ac, "error": msg})

    elapsed = time.time() - t0
    manifest = {
        "provenance": {
            "eval_dataset": args.eval,
            "composer_model": args.composer_model,
            "image_modality": args.image_modality,
            "families_per_class": args.families_per_class,
            "tiers": sorted(want_tiers),
            "note": "Mode-B corroboration frozen once; eval LOADS these (never composes at eval time). "
                    "BOTH arms are DETERMINISTIC (no LLM authoring -> refusal-immune + reproducible): "
                    "the image arm rasterizes the rendered value lines into a realistic carrier's "
                    "pixels behind a value-free citation; the text-arm control embeds the SAME value "
                    "lines in the page text under the SAME citation wrapper + ticket. Each record "
                    "pairs the two arms with their shared ground truth (forged value + landing "
                    "contract) so the ONLY difference between arms is the delivery channel (pixels vs "
                    "text). The composer model is retained only to resolve the ledger value; it does "
                    "NOT author any citation text.",
        },
        "frozen_count": len(frozen),
        "errors": errors,
        "skipped": skipped,
        "scenarios": frozen,
    }
    out_path = os.path.join(args.out_dir, "mode_b_frozen.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 68)
    print(f"Froze {len(frozen)}/{len(selected)} Mode-B scenarios "
          f"({len(errors)} failed, {len(skipped)} clean-skipped) in {elapsed:.1f}s")
    print(f"Wrote {out_path} + {len(frozen)} PNGs under {images_dir}")
    if skipped:
        for s in skipped:
            print(f"  ⊘ {s['scenario_id']} ({s['attack_class']}): {s['reason']}")
    if errors:
        for e in errors:
            print(f"  ✗ {e['scenario_id']}: {e['error']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
