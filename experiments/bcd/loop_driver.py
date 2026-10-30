#!/usr/bin/env python3
"""
loop_driver.py — B 组：闭环多循环学习曲线（v2，修正版）
=========================================================
v1 的设计错误：把 A2 做成"按历史合格率重分配采样预算"——那是**分布偏移**，
不是纠错。真正对应"LoRA 生成不稳定 + 同组不一致"的闭环反馈应该是
**失败即换种子重试（拒绝采样）**。

v2 三个 arm（单一变量：反馈用在哪）
------------------------------------
  A1  open-loop        均匀分层采样，单次尝试，验证器只做事后统计
  A2  rejection        均匀分层采样 + 失败重试（换 appearance seed，最多 R 次）
                       反馈用在"纠错"：不合格样本重新采样
  A3  full closed-loop 在 A2 基础上，用累积样本训练 yield 预测器，
                       把重试预算按预测合格率分配到各分层（学"哪里值得多试"）
                       + 按实测合格率重分配采样权重

每个 cycle = 一批目标点；每 round = 一次尝试。指标核心是
**每个 attempt 换来多少"通过验证"的样本**（yield per attempt），
以及**每个合格样本的 GPU 秒数**。

⚠️ 诚信提示：A2/A3 接受的样本是"通过验证器筛选"的子集，
   其分布 ≠ 生成器的原始输出分布。论文中必须写明这一选择偏差。

用法
----
    python3 loop_driver.py --arm A1 --cycles 4 --per-cycle 60 --out_B/A1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WS = ROOT / "0-workspace"
EVAL_DIR = ROOT / "experiments" / "new test"

for p in [str(HERE), str(WS), str(WS / "app"), str(WS / "generator"),
          str(WS / "loop"),
          str(WS / "validator"), str(WS / "validator" / "v6_quality"),
          str(WS / "validator" / "v7_consistency"), str(EVAL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import harness  # noqa: E402
from adapters import to_validator_sample  # noqa: E402
from eval_core import brisque_score, compute_alignment  # noqa: E402

DIST_BINS = harness.DISTANCES
WEATHERS = harness.WEATHERS
STRATA = [(d, w) for d in DIST_BINS for w in WEATHERS]
FLOOR_RATIO = 0.3


# ════════════════════════════════════════════════════════════════════════
# 采样策略
# ════════════════════════════════════════════════════════════════════════

def uniform_weights() -> dict:
    u = 1.0 / len(STRATA)
    return {s: u for s in STRATA}


def reallocate(weights: dict, pass_stats: dict, alpha: float = 2.0,
               beta: float = 1.0) -> dict:
    """实测合格率高的分层获得更多采样预算（带覆盖下限，防止坍塌到近场）。"""
    tot_p = sum(v[0] for v in pass_stats.values())
    tot_n = sum(v[1] for v in pass_stats.values())
    g = (tot_p + alpha) / (tot_n + 2 * alpha) if tot_n else 0.5
    raw = {}
    for s in STRATA:
        p, n = pass_stats.get(s, (0, 0))
        rate = (p + alpha) / (n + 2 * alpha) if n else g
        raw[s] = weights.get(s, 1.0 / len(STRATA)) * (max(rate / max(g, 1e-6), 1e-3) ** beta)
    floor = FLOOR_RATIO / len(STRATA)
    raw = {s: max(v, floor) for s, v in raw.items()}
    z = sum(raw.values())
    return {s: v / z for s, v in raw.items()}


# ════════════════════════════════════════════════════════════════════════
# 学习型 yield 预测器（A3）—— 只用生成前已知量，无信息泄漏
# ════════════════════════════════════════════════════════════════════════

def _featurize(rec: dict) -> np.ndarray:
    d = float(rec["distance"]) / 200.0
    tod = [1.0 if rec["time_of_day"] == t else 0.0 for t in harness.TIMES]
    wx = [1.0 if rec["weather"] == w else 0.0 for w in WEATHERS]
    return np.array([d, d * d] + tod + wx, dtype=np.float64)


class YieldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.n_train = 0

    def fit(self, recs: list, labels: list) -> None:
        y = np.asarray(labels, dtype=np.int64)
        if len(y) < 20 or len(set(y.tolist())) < 2:
            return
        X = np.stack([_featurize(r) for r in recs])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=3000)
        m.fit(self.scaler.transform(X), y)
        self.model, self.n_train = m, len(y)

    def prob(self, recs: list) -> np.ndarray:
        if self.model is None:
            return np.full(len(recs), 0.5)
        X = np.stack([_featurize(r) for r in recs])
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


# ════════════════════════════════════════════════════════════════════════
# 目标点规划
# ════════════════════════════════════════════════════════════════════════

def plan_cycle(weights: dict, n: int, rng: random.Random,
               used_bg: set, cycle: int) -> list:
    strata = list(weights.keys())
    p = np.array([weights[s] for s in strata], dtype=np.float64)
    p /= p.sum()
    rng_np = np.random.RandomState(rng.randrange(2 ** 31))

    quota = Counter()
    for _ in range(n):
        quota[strata[int(rng_np.choice(len(strata), p=p))]] += 1

    items, idx = [], 0
    for (dist, weather), k in quota.items():
        for _ in range(k):
            scene = harness.SCENES[idx % len(harness.SCENES)]
            items.append({
                "scene": scene,
                "background": _pick_bg(scene, rng, used_bg),
                "time_of_day": harness.TIMES[idx % len(harness.TIMES)],
                "weather": weather,
                "distance": dist,
                "action": harness.ACTIONS[idx % len(harness.ACTIONS)],
                "norm_u": round(rng.uniform(0.12, 0.88), 4),
                "norm_v": round(rng.uniform(0.10, 0.68), 4),
            })
            idx += 1
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["trajectory"] = [{"t": 0, "action": it["action"],
                             "distance": it["distance"],
                             "norm_u": it["norm_u"], "norm_v": it["norm_v"]}]
        it["gid"] = f"k{cycle}_{i:04d}"
    return items


def _pick_bg(scene: str, rng: random.Random, used_bg: set) -> str:
    files = harness.pool_for(scene)
    for _ in range(60):
        c = files[rng.randrange(len(files))]
        if str(c) not in used_bg:
            used_bg.add(str(c))
            return str(c)
    return str(files[rng.randrange(len(files))])


# ════════════════════════════════════════════════════════════════════════
# 验证 + 指标
# ════════════════════════════════════════════════════════════════════════

def build_pipeline():
    from validator_pipeline import ValidatorPipeline
    from quality_validator import QualityValidator
    from consistency_validator import ConsistencyValidator
    from v8_trajectory_validator import TrajectoryValidator
    from v9_detection_validator import DetectionValidator
    from failure_buffer import FailureBuffer
    s3 = DetectionValidator(model_path=str(ROOT / "0-model" / "yolov8x.pt"),
                            device="cuda:0")
    return ValidatorPipeline(
        s0=QualityValidator(), s1=ConsistencyValidator(),
        s2=TrajectoryValidator(), s3=s3,
        failure_buffer=FailureBuffer(path=str(HERE / "failure_log.jsonl"),
                                     threshold=50))


def _gid_of(sample_id: str) -> str:
    """从 sample_id 反推 gid。

    harness.generate_batch() 不回传 gid（其条目 schema 里没有该字段），
    因此 score_records 拿到的 e["gid"] 恒为 None。若不还原，
    `accepted.setdefault(None, x)` 会把整批已接受样本塌缩成 1 条，
    且 status 字典键全为 None 导致下一轮重试名单永远不收敛。
    sample_id 形如 "k{cycle}_{i:04d}_r{rnd}" → gid 为 "k{cycle}_{i:04d}"。
    """
    return sample_id.rsplit("_r", 1)[0]


def score_records(pipe, cycle_dir: Path, entries: list) -> list[dict]:
    out = []
    for e in entries:
        s = to_validator_sample(e, cycle_dir)
        r = pipe.validate(s)
        H, W = s["rgb"].shape[:2]
        rec = {
            "sample_id": e["sample_id"],
            "gid": e.get("gid") or _gid_of(e["sample_id"]),
            "distance": e["distance"], "weather": e["weather"],
            "time_of_day": e["time_of_day"], "scene": e["scene"],
            "passed": bool(r.passed), "failure_code": r.failure_code,
            "score": round(float(r.total_score), 4),
            "attempt_seed": e.get("gen_seed"),
        }
        if e.get("bbox"):
            al = compute_alignment(np.asarray(e["bbox"], dtype=np.float64), W, H,
                                   e["nu"], e["nv"], float(e["target_px"]))
            rec["pos_err_px"] = round(float(al.get("pos_err_px", np.nan)), 3)
            rec["iou"] = round(float(al.get("iou", np.nan)), 4)
        rec["brisque"] = round(float(brisque_score(s["rgb"])), 2)
        out.append(rec)
    return out


# ════════════════════════════════════════════════════════════════════════
# 单 cycle：带拒绝采样的多轮生成
# ════════════════════════════════════════════════════════════════════════

def run_cycle(engine, pipe, arm: str, items: list, cdir: Path,
              max_retries: int, cycle: int, predictor, seed_base: int,
              skip_cnet: bool) -> dict:
    """round 0 = 全部目标点；后续 round 只重试上一轮失败的。

    A3 的差异：重试预算按预测 yield 分配——预测必失败的组不浪费重试额度。
    """
    pending = list(items)
    attempts = 0
    accepted: dict[str, dict] = {}
    round_log = []
    all_scored: list[dict] = []
    t0 = time.time()

    for rnd in range(max_retries + 1):
        if not pending:
            break
        batch = []
        for i, it in enumerate(pending):
            b = dict(it)
            b["sample_id"] = f"{it['gid']}_r{rnd}"
            b["trajectory"] = [dict(it["trajectory"][0])]
            batch.append(b)

        # 每轮换 appearance seed → 同一"目标点"但无人机外观不同（拒绝采样）
        prev = engine.override_gen_seed
        engine.override_gen_seed = int(seed_base + cycle * 1_000_003 + rnd * 7_919)
        try:
            entries = harness.generate_batch(
                engine, batch, cdir, skip_cnet=skip_cnet, flush_every=10,
                start_session=100000 + cycle * 10_000 + rnd * 1_000)
        finally:
            engine.override_gen_seed = prev
        attempts += len(entries)

        scored = score_records(pipe, cdir, entries)
        for x in scored:
            x["round"] = rnd
        all_scored.extend(scored)
        ok = [x for x in scored if x["passed"]]
        bad = [x for x in scored if not x["passed"]]
        for x in ok:
            accepted.setdefault(x["gid"], x)

        round_log.append({
            "round": rnd, "attempts": len(entries),
            "passed": len(ok), "failed": len(bad),
            "pass_rate": round(len(ok) / max(len(entries), 1), 4),
            "failure_hist": dict(Counter(x["failure_code"] or "PASS" for x in scored)),
        })
        print(f"  [cycle {cycle}] round {rnd}: {len(entries)} attempts → "
              f"{len(ok)} pass ({len(ok)/max(len(entries),1)*100:.1f}%)",
              flush=True)

        # 下一轮要重试谁
        status = {x["gid"]: x for x in scored}
        pending = [it for it in pending
                   if not status.get(f"{it['gid']}_r{rnd}", {}).get("passed")]

        # A3：按预测 yield 裁剪重试名单（预测无望的不再浪费 GPU）
        if arm == "A3" and predictor is not None and predictor.model is not None \
                and pending and rnd < max_retries:
            p = predictor.prob(pending)
            keep = [it for it, pi in zip(pending, p) if pi >= 0.15]
            if keep and len(keep) < len(pending):
                print(f"  [cycle {cycle}] A3 curriculum: retry set "
                      f"{len(pending)} → {len(keep)} (预测无望者放弃)", flush=True)
                pending = keep
            elif not keep:
                pending = pending[: max(1, len(pending) // 3)]

    gen_sec = time.time() - t0
    acc = list(accepted.values())
    res = {
        "cycle": cycle,
        "n_targets": len(items),
        "attempts": attempts,
        "n_accepted": len(acc),
        "yield_per_attempt": round(len(acc) / max(attempts, 1), 4),
        "acceptance_over_targets": round(len(acc) / max(len(items), 1), 4),
        "gen_sec": round(gen_sec, 1),
        "sec_per_accepted": round(gen_sec / max(len(acc), 1), 1),
        "rounds": round_log,
        "per_sample": all_scored,
    }
    if acc:
        res["brisque_mean"] = round(float(np.mean([x["brisque"] for x in acc])), 2)
        res["iou_mean"] = round(float(np.mean([x["iou"] for x in acc if x.get("iou")])), 4)
        res["failure_hist_accepted"] = None
    return res


# ════════════════════════════════════════════════════════════════════════
# 主
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A1", "A2", "A3"])
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--per-cycle", type=int, default=60)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cnet", action="store_true")
    a = ap.parse_args()

    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    max_retries = 0 if a.arm == "A1" else a.max_retries

    print(f"[B] arm={a.arm} cycles={a.cycles} per_cycle={a.per_cycle} "
          f"max_retries={max_retries}", flush=True)
    engine = harness.load_engine()
    pipe = build_pipeline()

    weights = uniform_weights()
    used_bg: set = set()
    rng = random.Random(a.seed)
    predictor = YieldPredictor() if a.arm == "A3" else None
    hist: list[dict] = []

    curve = {"arm": a.arm, "cycles": a.cycles, "per_cycle": a.per_cycle,
             "max_retries": max_retries, "seed": a.seed, "results": [],
             "note": ("A2/A3 接受的样本经过验证器筛选，分布 ≠ 生成器原始输出分布；"
                      "这是拒绝采样的固有选择偏差，论文须写明。"
                      "【S9 循环论证】验证器末级为 DetectionValidator(YOLOv8x, conf_thr=0.25)，"
                      "判定规则为「存在未检测到的 GT 即 FAIL」。A1 实测 S9_UNDETECTABLE 占失败"
                      "总数约 70%，即闭环信号主要由「该图能否被 YOLOv8x 检出」驱动，"
                      "而非通用图像质量。因此："
                      "(a) 本组数据不得用于评估或报告 YOLOv8x 及其同族模型的召回率，"
                      "否则构成验证器与评测器同一的循环论证；"
                      "(b) 下游检测可用性只能以留出的真实数据评估（见 4.5.8 节）；"
                      "(c) 论文表述应说明闭环优化的是「面向既定检测器的可检出性」，"
                      "不可泛化为「生成质量提升」。")}

    for k in range(a.cycles):
        cdir = out / f"cycle_{k}"
        cdir.mkdir(parents=True, exist_ok=True)
        w_used = dict(weights)
        items = plan_cycle(weights, a.per_cycle, rng, used_bg, k)
        print(f"[B] cycle {k}: {len(items)} targets", flush=True)

        res = run_cycle(engine, pipe, a.arm, items, cdir, max_retries, k,
                        predictor, a.seed * 99991, a.skip_cnet)
        res["weights_used"] = {f"{s[0]}|{s[1]}": round(v, 5) for s, v in w_used.items()}

        # ── 反馈：累积历史 → 训练预测器（A3）+ 重分配采样权重（A2/A3）──
        ps = res.pop("per_sample")
        hist.extend(ps)
        r0 = [x for x in ps if x.get("round") == 0]
        per_stratum = defaultdict(lambda: [0, 0])
        for x in r0:
            st = (x["distance"], x["weather"])
            per_stratum[st][1] += 1
            per_stratum[st][0] += int(x["passed"])
        res["round0_pass_rate"] = round(
            sum(1 for x in r0 if x["passed"]) / max(len(r0), 1), 4)
        res["round0_failure_hist"] = dict(Counter(
            x["failure_code"] or "PASS" for x in r0))

        if predictor is not None and a.arm == "A3":
            before = predictor.n_train
            predictor.fit(hist, [int(x["passed"]) for x in hist])
            if predictor.n_train != before:
                print(f"  [cycle {k}] yield predictor trained on "
                      f"{predictor.n_train} samples", flush=True)

        if a.arm in ("A2", "A3"):
            weights = reallocate(weights, dict(per_stratum))
        else:
            weights = uniform_weights()

        curve["results"].append(res)
        cum_att = sum(x["attempts"] for x in curve["results"])
        cum_acc = sum(x["n_accepted"] for x in curve["results"])
        curve["cumulative"] = {"attempts": cum_att, "accepted": cum_acc,
                               "yield": round(cum_acc / max(cum_att, 1), 4)}
        (out / "curve.json").write_text(
            json.dumps(curve, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[B] cycle {k}: accepted {res['n_accepted']}/{res['n_targets']} "
              f"attempts={res['attempts']} yield={res['yield_per_attempt']} "
              f"{res['sec_per_accepted']}s/样本", flush=True)
        print(f"[B] cumulative: {cum_acc} accepted / {cum_att} attempts "
              f"= {curve['cumulative']['yield']}", flush=True)

    print(f"[B] done → {out/'curve.json'}", flush=True)


if __name__ == "__main__":
    main()
