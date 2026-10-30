#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滑动排除队列（v7.5）行为探针 —— 为论文 §3.3.1 的机制声明提供可复现证据。

论文声称：*a sliding exclusion queue keeps the most recently used backgrounds out of the
eligible set, so the pool is traversed without repetition until a tier is exhausted,
after which the queue recycles its entries.*

本脚本对若干 (scene, tod, weather) 组合连续调用 `search_background()` N 次，统计
（a）命中不同文件数、（b）首次重复出现的位置，与上述声明逐条对照，并把原始记录落盘。

用法：
    cd 0-workspace/app && python3 probe_exclude_queue.py [--n 30] [--json out.json]
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CASES = [
    ("mountain", "dusk", "clear", 30),      # 小 tier（约 14 张）→ 应在该层耗尽后开始回收
    ("urban", "noon", "overcast", 30),
    ("desert", "dawn", "clear", 30),
    ("forest", "night", "clear", 30),
]


def probe(scene, tod, weather, n):
    import background_searcher as bs
    bs._RECENT_BG.clear()                    # 每例从干净状态开始
    picks, tiers, tiers_extra, cands = [], [], [], []
    for _ in range(n):
        r = bs.search_background({"scene_type": scene, "time_of_day": tod,
                                  "weather": weather}, session_id=0)
        if not r.get("ok"):
            raise RuntimeError(r)
        picks.append(Path(r["source_path"]).name)
        tiers.append(r.get("match_level"))
        tiers_extra.append(r.get("pool_size"))
        cands.append((r.get("match_level"), r.get("candidates"), r.get("pool_size")))
    first_rep = next((i + 1 for i, v in enumerate(picks)
                      if picks.index(v) != i), None)
    return {
        "case": f"{scene}/{tod}/{weather}",
        "calls": n,
        "distinct": len(set(picks)),
        "first_repeat_at": first_rep,        # None = 全程无重复
        "exclude_window": bs.EXCLUDE_WINDOW,
        "window": bs.EXCLUDE_WINDOW,
        "tier_size": max((int(c[1]) for c in cands if c[1]), default=None),
        "exclude_keep_min": bs.EXCLUDE_KEEP_MIN,
        "tiers_used": sorted({str(t) for t in tiers}),
        "pool_sizes": sorted({int(t) for t in tiers_extra}) if tiers_extra else [],
        "picks": picks,
        "pick_counts": dict(Counter(picks).most_common(5)),
        "tier_candidates": cands[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="覆盖各例的调用次数")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    out = {"generated": datetime.now().isoformat(timespec="seconds"),
           "env": {"BG_EXCLUDE_WINDOW": os.environ.get("BG_EXCLUDE_WINDOW"),
                   "BG_EXCLUDE_KEEP_MIN": os.environ.get("BG_EXCLUDE_KEEP_MIN")},
           "results": []}
    ok = True
    for scene, tod, weather, n in CASES:
        r = probe(scene, tod, weather, a.n or n)
        out["results"].append(r)
        # 判据（与论文 §3.3.1 的措辞一一对应，是一个析取式声明）：
        #   ① 该层候选数 ≥ 窗口 → 任意 pick[i] 不得出现在其前 window 次之中
        #   ② 该层候选数 < 窗口 → 必须无重复地遍历至耗尽（首次重复 == tier_size+1）
        w = r["window"]
        t = r["tier_size"]
        viol = [(i + 1, r["picks"][i]) for i in range(len(r["picks"]))
                if r["picks"][i] in r["picks"][max(0, i - w):i]]
        if t and t < w:
            verdict = "OK" if (r["first_repeat_at"] == t + 1) else "FAIL"
            note = f"tier{t}<窗口{w} → 遍历至耗尽"
        else:
            verdict = "OK" if not viol else "FAIL"
            note = f"tier{t}≥窗口{w} → 窗口内不重复"
        r["window"] = w
        r["window_violations"] = viol
        r["rule"] = note
        r["verdict"] = verdict
        ok &= verdict == "OK"
        print(f"[{verdict}] {r['case']:<24} tier={t:>3} distinct={r['distinct']:>3} "
              f"首次重复={r['first_repeat_at']} 窗口内违规={len(viol)}  ({note})")
    out["all_pass"] = ok
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n记录已落盘：{a.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
