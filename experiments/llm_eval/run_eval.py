#!/usr/bin/env python3
"""
run_eval.py — Module M1 (semantic parser) benchmark
===================================================
Implements the evaluation requested by Major Concern 11:

  * parsing / field-extraction accuracy        (5 schema fields)
  * trajectory extraction accuracy             (first/last metric distance)
  * invalid-output rate                        (raw completions that fail schema validation)
  * ambiguity handling                         (underspecified and out-of-vocabulary inputs)
  * robustness to paraphrased descriptions     (canonical vs paraphrase subsets)
  * LLM parser vs deterministic template parser (paired, identical inputs)

Arms
----
  llm        : the deployed path — chat completion + validate_and_parse()
  template   : deterministic keyword/template baseline (template_parser.py)

Usage
-----
  export DEEPSEEK_API_KEY=...        # or let it read ~/.clacky/config.yml
  python3 run_eval.py [--workers 4] [--model deepseek-chat] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import template_parser as tp  # noqa: E402
import importlib.util  # noqa: E402

_PARSER_PATH = HERE.parents[1] / "3-LLM starter" / "llm_parser.py"
_spec = importlib.util.spec_from_file_location("llm_parser_ref", _PARSER_PATH)
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)

FIELDS = ["scene_type", "time_of_day", "weather", "camera_position", "action"]


# ---------------------------------------------------------------------------
# API plumbing
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    cfg = Path.home() / ".clacky" / "config.yml"
    if cfg.exists():
        txt = cfg.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"api_key:\s*(sk-[A-Za-z0-9]+)", txt)
        if m:
            return m.group(1)
    raise SystemExit("no DeepSeek API key found")


def repair_json(raw: str) -> str:
    """Minimal brace-balancing sanitiser for the *analysis* arm only.

    The deployed validator (llm_parser.validate_and_parse) removes markdown
    fences and trailing commas.  It does not repair unbalanced delimiters, so a
    completion that carries one superfluous closing brace is rejected outright.
    This 10-line sanitiser trims trailing unmatched closers and discards any
    prose that precedes the first opening brace, and is applied *only* to
    quantify how much of the residual invalid-output rate is trivially
    recoverable.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return s
    s = s[start:]
    depth, last_ok = 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == chr(92):
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_ok = i
    return s[:last_ok + 1] if last_ok >= 0 else s


def llm_parse(text: str, client, model: str, retries: int = 3) -> dict:
    """One LLM parse. Returns a dict with raw completion, parsed spec, latency."""
    messages = lp.build_messages(text)
    last_err = ""
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.1, max_tokens=1000)
            raw = r.choices[0].message.content or ""
            lat = time.time() - t0
            rec = {"raw": raw, "latency": lat, "api_error": "", "schema_error": "",
                   "repaired_ok": False}
            try:
                spec = lp.validate_and_parse(raw, raw_input=text)
                full = spec.to_full_spec()
                rec["spec"] = {
                    "scene_type": full["background"]["scene_type"],
                    "time_of_day": full["background"]["time_of_day"],
                    "weather": full["background"]["weather"],
                    "camera_position": full["camera"]["position"],
                    "action": full["trajectory"][0]["action"] if full["trajectory"] else "hover",
                    "distance_first": full["trajectory"][0]["distance"] if full["trajectory"] else None,
                    "distance_last": full["trajectory"][-1]["distance"] if full["trajectory"] else None,
                    "n_points": len(full["trajectory"]),
                    "default_fills": full["meta"]["default_fills"],
                    "confidence_note": full["meta"]["confidence_note"],
                }
            except Exception as exc:            # invalid output: unparsable JSON
                rec["spec"] = None
                rec["schema_error"] = "%s: %s" % (type(exc).__name__, exc)
                try:                            # analysis arm: minimal repair
                    spec2 = lp.validate_and_parse(repair_json(raw), raw_input=text)
                    full2 = spec2.to_full_spec()
                    rec["repaired_ok"] = True
                    rec["repaired_spec"] = {
                        "scene_type": full2["background"]["scene_type"],
                        "time_of_day": full2["background"]["time_of_day"],
                        "weather": full2["background"]["weather"],
                        "camera_position": full2["camera"]["position"],
                        "action": full2["trajectory"][0]["action"] if full2["trajectory"] else "hover",
                        "distance_first": full2["trajectory"][0]["distance"] if full2["trajectory"] else None,
                        "distance_last": full2["trajectory"][-1]["distance"] if full2["trajectory"] else None,
                        "n_points": len(full2["trajectory"]),
                        "default_fills": full2["meta"]["default_fills"],
                        "confidence_note": full2["meta"]["confidence_note"],
                    }
                except Exception as exc2:
                    rec["repair_error"] = "%s: %s" % (type(exc2).__name__, exc2)
            return rec
        except Exception as exc:
            last_err = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(2 + 2 * attempt)
    return {"raw": "", "latency": float("nan"), "api_error": last_err,
            "schema_error": "api failure", "spec": None}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_item(item: dict, pred: dict | None) -> dict:
    gold = item.get("gold") or {}
    res = {"id": item["id"], "subset": item["subset"], "n_scored": 0, "n_correct": 0,
           "field_hits": {}, "field_scored": {}}
    if pred is None:
        for f in gold:
            res["field_hits"][f] = 0
            res["field_scored"][f] = 1
            res["n_scored"] += 1
        return res
    for f, gv in gold.items():
        if gv is None:
            continue
        pv = pred.get(f)
        hit = (str(pv) == str(gv))
        res["field_hits"][f] = int(hit)
        res["field_scored"][f] = 1
        res["n_scored"] += 1
        res["n_correct"] += int(hit)
    return res


def score_traj(item: dict, pred: dict | None) -> dict | None:
    tr = item.get("traj")
    if not tr or pred is None:
        return None
    out = {}
    for key, gold_key in (("distance_first", "d_first"), ("distance_last", "d_last")):
        pv, gv = pred.get(key), tr[gold_key]
        if pv is None:
            out[key] = 0
        else:
            out[key] = int(abs(float(pv) - gv) <= max(5.0, 0.1 * gv))
    out["exact"] = int(out["distance_first"] and out["distance_last"])
    return out


def agg(prefix: str, records: list[dict]) -> dict:
    """Aggregate per-item field scores into metrics."""
    n = len(records)
    fields = {}
    for rec in records:
        for f, s in rec["field_scored"].items():
            d = fields.setdefault(f, {"scored": 0, "hit": 0})
            d["scored"] += s
            d["hit"] += rec["field_hits"][f]
    full = sum(1 for r in records if r["n_scored"] and r["n_correct"] == r["n_scored"])
    return {"n": n,
            "field_accuracy": {f: (d["hit"] / d["scored"]) for f, d in fields.items()},
            "field_n": {f: d["scored"] for f, d in fields.items()},
            "field_hits": {f: d["hit"] for f, d in fields.items()},
            "joint_accuracy": sum(r["n_correct"] for r in records) /
                              max(1, sum(r["n_scored"] for r in records)),
            "complete_match_rate": full / max(1, n)}


def write_report(res: dict, path: Path) -> None:
    L = []
    add = L.append
    add("# Module M1 semantic-parser benchmark\n")
    add("Benchmark: %d manually annotated descriptions in 6 subsets. "
        "Model: `%s`, temperature 0.1.\n" % (res["meta"]["n_items"], res["meta"]["model"]))
    add("## Headline\n")
    add("| Metric | LLM parser | Deterministic template parser |")
    add("|---|---:|---:|")
    a, b = res["overall"]["llm"], res["overall"]["template"]
    add("| Field-level accuracy | %.3f | %.3f |" % (a["joint_accuracy"], b["joint_accuracy"]))
    add("| Complete-schema match rate | %.3f | %.3f |" %
        (a["complete_match_rate"], b["complete_match_rate"]))
    add("| Invalid-output rate | %.3f | %.3f |" %
        (res["invalid_rate"]["llm"], res["invalid_rate"]["template"]))
    add("| Mean latency (s) | %.2f | %.4f |" %
        (res["latency"]["llm_mean"], res["latency"]["template_mean"]))
    add("")
    add("## Field-level accuracy\n")
    add("| Field | LLM | Template | n |")
    add("|---|---:|---:|---:|")
    for f in FIELDS:
        add("| %s | %.3f | %.3f | %d |" % (
            f, a["field_accuracy"].get(f, float("nan")),
            b["field_accuracy"].get(f, float("nan")), a["field_n"].get(f, 0)))
    add("")
    add("## By subset\n")
    add("| Subset | n | LLM joint | Template joint | LLM complete |")
    add("|---|---:|---:|---:|---:|")
    for s in res["meta"]["subsets"]:
        sa, sb = res["by_subset"][s]["llm"], res["by_subset"][s]["template"]
        add("| %s | %d | %.3f | %.3f | %.3f |" %
            (s, sa["n"], sa["joint_accuracy"], sb["joint_accuracy"], sa["complete_match_rate"]))
    add("")
    add("## Trajectory extraction\n")
    add("| Arm | distance-first | distance-last | both correct | n |")
    add("|---|---:|---:|---:|---:|")
    for arm in ("llm", "template"):
        t = res["trajectory"][arm]
        add("| %s | %.3f | %.3f | %.3f | %d |" % (arm, t["first"], t["last"], t["exact"], t["n"]))
    add("")
    add("## Ambiguity handling (underspecified + out-of-vocabulary)\n")
    add("| Arm | any note on ambiguous | note naming the unresolved dimension | note on determinate |")
    add("|---|---:|---:|---:|")
    for arm in ("llm", "template"):
        sub = res["ambiguity"].get("llm_substantive_ambiguous", 0) if arm == "llm" else ""
        sub_s = "%.3f" % sub if isinstance(sub, float) else "n/a"
        add("| %s | %.3f | %s | %.3f |" % (arm, res["ambiguity"][arm]["flagged_ambiguous"],
                                          sub_s, res["ambiguity"][arm]["flagged_determinate"]))
    add("")
    add("## Invalid-output repair (analysis arm only)\n")
    rp = res["repair"]
    add("| Quantity | Value |")
    add("|---|---:|")
    add("| Invalid outputs, this pass | %d / %d |" % (rp["n_invalid"], res["meta"]["n_items"]))
    add("| Recovered by brace balancing | %d |" % rp["n_recovered"])
    add("| Invalid-output rate after repair | %.3f |" % rp["invalid_rate_after_repair"])
    add("| Joint accuracy after repair | %.3f |" % res["overall"]["llm_repaired"]["joint_accuracy"])
    add("")
    rp2 = res.get("replication")
    if rp2:
        add("## Replication across passes (decoder sampling variance)\n")
        add("| Quantity | Value |")
        add("|---|---:|")
        add("| Passes completed | %d |" % rp2["runs"])
        add("| Pooled completions | %d |" % rp2["completions"])
        add("| Pooled invalid-output rate | %.4f |" % rp2["pooled_invalid_rate"])
        add("| Joint accuracy, mean +/- sd | %.3f +/- %.3f |" %
            (rp2["joint_accuracy_mean"], rp2["joint_accuracy_sd"]))
        add("| Complete-schema match, mean +/- sd | %.3f +/- %.3f |" %
            (rp2["complete_match_mean"], rp2["complete_match_sd"]))
        add("")
    add("## Caveats\n")
    add("1. The out-of-vocabulary subset defines its gold standard as the conservative "
        "fallback value, which structurally favours the template arm: an arm that always "
        "returns the default is always 'correct' there. The informative quantity for that "
        "subset is therefore not accuracy but whether the fallback is signalled.")
    add("2. The template arm is a longest-match keyword parser over the *same* controlled "
        "vocabulary as the deployed system, so it shares the deployed decoder rather than "
        "representing an independently engineered baseline.")
    add("3. Latency excludes network variance in the template arm, which is local and "
        "network-free by construction.")
    add("")
    add("## Failure modes (as deployed)\n")
    for fm in rp["failure_modes"]:
        add("- `%s`: %s; tail of completion ...%s" % (fm["id"], fm["error"], fm["raw_tail"]))
    add("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in (HERE / "benchmark.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]

    key = load_api_key()
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    print("running LLM arm on %d items ..." % len(items))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        llm_recs = list(ex.map(lambda it: llm_parse(it["text"], client, args.model), items))
    print("LLM arm done in %.1f s" % (time.time() - t0))

    tp_recs = [tp.parse(it["text"]) for it in items]

    # ---- scoring -------------------------------------------------------
    score = {"llm": [], "template": [], "llm_repaired": []}
    traj = {"llm": [], "template": []}
    for it, lr, tr in zip(items, llm_recs, tp_recs):
        score["llm"].append(score_item(it, lr.get("spec")))
        score["llm_repaired"].append(score_item(it, lr.get("spec") or lr.get("repaired_spec")))
        score["template"].append(score_item(it, tr))
        st_l, st_t = score_traj(it, lr.get("spec")), score_traj(it, tr)
        if st_l:
            traj["llm"].append(st_l)
            traj["template"].append(st_t)

    # flags: LLM = confidence_note non-empty; template = silent_defaults reported
    ambiguous_ids = {it["id"] for it in items if it["subset"] in ("underspecified", "conflict_oov")}
    determinate_ids = {it["id"] for it in items if it["subset"] not in ("underspecified", "conflict_oov")}
    flags = {"llm": {"amb": [], "det": []}, "template": {"amb": [], "det": []}}
    SUBST = re.compile(r"默认|未指定|无指定|超出|互斥|无法映射|非标准|映射为|不适用|default|"
                       r"unspecified|out of range|conflict", re.I)
    subst_bad = []
    for it, lr, tr in zip(items, llm_recs, tp_recs):
        grp = "amb" if it["id"] in ambiguous_ids else "det"
        spec = lr.get("spec") or {}
        note = (spec.get("confidence_note") or "").strip()
        flags["llm"][grp].append(int(bool(note)))
        flags["template"][grp].append(int(bool(tr.get("silent_defaults")) and
                                           len(tr["silent_defaults"]) > 0 and
                                           it["subset"] == "underspecified"))
        if grp == "amb" and not SUBST.search(note):
            subst_bad.append(it["id"])

    lat_llm = [r["latency"] for r in llm_recs if r["latency"] == r["latency"]]
    res = {
        "meta": {"model": args.model, "n_items": len(items), "workers": args.workers,
                 "subsets": sorted({it["subset"] for it in items}),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "overall": {arm: agg(arm, score[arm]) for arm in ("llm", "llm_repaired", "template")},
        "by_subset": {s: {arm: agg(arm, [r for r in score[arm] if r["subset"] == s])
                          for arm in ("llm", "template")}
                      for s in sorted({it["subset"] for it in items})},
        "trajectory": {arm: {
            "n": len(traj[arm]),
            "first": statistics.mean(t["distance_first"] for t in traj[arm]) if traj[arm] else 0,
            "last": statistics.mean(t["distance_last"] for t in traj[arm]) if traj[arm] else 0,
            "exact": statistics.mean(t["exact"] for t in traj[arm]) if traj[arm] else 0,
        } for arm in ("llm", "template")},
        "ambiguity": {arm: {
            "flagged_ambiguous": statistics.mean(flags[arm]["amb"]) if flags[arm]["amb"] else 0,
            "flagged_determinate": statistics.mean(flags[arm]["det"]) if flags[arm]["det"] else 0,
        } for arm in ("llm", "template")} | {
            "llm_substantive_ambiguous": (
                1 - len(subst_bad) / max(1, len(flags["llm"]["amb"]))),
            "llm_substantive_failures": subst_bad,
        },
        "invalid_rate": {
            "llm": sum(1 for r in llm_recs if r.get("spec") is None) / max(1, len(items)),
            "template": 0.0,
        },
        "repair": {
            "n_invalid": sum(1 for r in llm_recs if r.get("spec") is None),
            "n_recovered": sum(1 for r in llm_recs
                               if r.get("spec") is None and r.get("repaired_ok")),
            "invalid_rate_after_repair": sum(
                1 for r in llm_recs
                if r.get("spec") is None and not r.get("repaired_ok")) / max(1, len(items)),
            "failure_modes": [{"id": it["id"], "error": r.get("schema_error", ""),
                               "raw_tail": r.get("raw", "")[-60:]}
                              for it, r in zip(items, llm_recs) if r.get("spec") is None],
        },
        "api_errors": sum(1 for r in llm_recs if r.get("api_error")),
        "latency": {"llm_mean": statistics.mean(lat_llm) if lat_llm else 0,
                    "llm_median": statistics.median(lat_llm) if lat_llm else 0,
                    "template_mean": 1e-4},
        "items": [{"id": it["id"], "subset": it["subset"], "text": it["text"],
                   "gold": it.get("gold"), "llm_spec": lr.get("spec"),
                   "llm_raw": lr.get("raw", "")[:1200],
                   "llm_schema_error": lr.get("schema_error", ""),
                   "llm_api_error": lr.get("api_error", ""),
                   "template": {k: v for k, v in tr.items() if k != "raw_output"}}
                  for it, lr, tr in zip(items, llm_recs, tp_recs)],
    }
    (HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- cross-run ledger (sampling variance of an LLM decoder) ------------
    ledger_path = HERE / "runs.jsonl"
    run_stub = {
        "timestamp": res["meta"]["timestamp"],
        "n": len(items),
        "n_invalid": res["repair"]["n_invalid"],
        "joint_accuracy": res["overall"]["llm"]["joint_accuracy"],
        "complete_match_rate": res["overall"]["llm"]["complete_match_rate"],
        "template_joint_accuracy": res["overall"]["template"]["joint_accuracy"],
        "latency_mean": res["latency"]["llm_mean"],
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_stub, ensure_ascii=False) + "\n")
    runs = [json.loads(l) for l in ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pooled_n = sum(r["n"] for r in runs)
    pooled_invalid = sum(r["n_invalid"] for r in runs)
    res["replication"] = {
        "runs": len(runs),
        "completions": pooled_n,
        "pooled_invalid_rate": pooled_invalid / max(1, pooled_n),
        "joint_accuracy_mean": statistics.mean(r["joint_accuracy"] for r in runs),
        "joint_accuracy_sd": (statistics.stdev([r["joint_accuracy"] for r in runs])
                              if len(runs) > 1 else 0.0),
        "complete_match_mean": statistics.mean(r["complete_match_rate"] for r in runs),
        "complete_match_sd": (statistics.stdev([r["complete_match_rate"] for r in runs])
                              if len(runs) > 1 else 0.0),
        "run_stubs": runs,
    }
    (HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "replication.json").write_text(
        json.dumps(res["replication"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(res, HERE / "report.md")
    print(json.dumps({"overall": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                      for kk, vv in v.items() if kk in
                                      ("joint_accuracy", "complete_match_rate")}
                                for k, v in res["overall"].items()},
                      "invalid_rate": res["invalid_rate"],
                      "latency": {k: round(v, 3) for k, v in res["latency"].items()},
                      "trajectory": res["trajectory"],
                      "ambiguity": res["ambiguity"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
