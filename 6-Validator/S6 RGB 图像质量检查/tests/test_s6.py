"""
S6 端到端测试
==============

1. 单元测试：IR sanity 三检、S6 判定逻辑分支。
2. 集成测试：真实帧 vs ControlNet 生成图，跑 S6 判定，输出对比统计。

运行：
  python3 test_s0.py                     # 全部
  python3 test_s0.py --frames <dir> --gen <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from rgb_quality import RGBQualityChecker         # noqa: E402
from ir_sanity import IRSanityChecker             # noqa: E402
from quality_validator import QualityValidator, S6Verdict  # noqa: E402

CODE_DIR = Path(__file__).parent.parent / "code"
CONFIG = Path(__file__).parent.parent / "config" / "s6_thresholds.json"


# ============================================================================
# 单元测试
# ============================================================================
def unit_tests():
    print("=" * 60)
    print("单元测试")
    print("=" * 60)
    ir = IRSanityChecker()

    # IR 三检
    flat = np.full((256, 256), 128, np.uint8)
    r = ir.check(flat)
    assert "IR_FLAT_OUTPUT" in r.warnings, "全灰应触发 IR_FLAT_OUTPUT"
    assert r.passed, "IR 线永不判 fail"
    print("  [OK] IR 全灰 → IR_FLAT_OUTPUT + passed")

    x = np.arange(256)
    stripe = np.tile((127 + 100 * np.sin(2 * np.pi * x / 8)).astype(np.uint8),
                     (256, 1))
    r = ir.check(stripe)
    assert "IR_FFT_MIDBAND_ARTIFACT" in r.warnings, "条纹应触发 FFT 告警"
    print("  [OK] IR 条纹 → IR_FFT_MIDBAND_ARTIFACT")

    over = np.zeros((64, 64), np.float32); over[0, 0] = 300
    r = ir.check(over)
    assert "IR_PIXEL_OUT_OF_RANGE" in r.warnings
    print("  [OK] IR 溢出 → IR_PIXEL_OUT_OF_RANGE")

    # S6 判定分支：构造合成图
    v = QualityValidator(brisque_thr=999)  # RGB 必过
    rgb = np.random.randint(0, 255, (128, 128, 3), np.uint8)
    res = v.validate({"id": "t", "rgb": rgb, "ir": flat})
    assert res.verdict == S6Verdict.PASS_WITH_WARNING
    print("  [OK] RGB合格+IR全灰 → PASS_WITH_WARNING")

    v2 = QualityValidator(brisque_thr=-1)  # RGB 必挂
    res = v2.validate({"id": "t", "rgb": rgb})
    assert res.verdict == S6Verdict.FAIL and res.passed is False
    print("  [OK] RGB不合格 → FAIL")

    res = v.validate({"id": "t", "rgb": None})
    assert res.reason == "S6_RGB_MISSING"
    print("  [OK] 缺 RGB → S6_RGB_MISSING")
    print("  单元测试全部通过\n")


# ============================================================================
# 集成测试：真实 vs 生成
# ============================================================================
def _scan(d, exts=("*.jpg", "*.jpeg", "*.png"), n=0):
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, "**", e), recursive=True))
    files.sort()
    return files[:n] if n > 0 else files


def integration_tests(frames_dir, gen_dir, n_real=50):
    print("=" * 60)
    print("集成测试：真实帧 vs 生成图")
    print("=" * 60)

    if CONFIG.exists():
        v = QualityValidator.from_config(str(CONFIG))
        with open(CONFIG) as f:
            thr = json.load(f)["rgb"]["brisque_thr"]
        print(f"  加载校准阈值 brisque_thr = {thr}")
    else:
        v = QualityValidator(brisque_thr=96.9)
        thr = 96.9
        print(f"  用默认阈值 {thr}（未找到校准配置）")

    def run(files, tag):
        recs = []
        for fp in files:
            im = cv2.imread(fp)
            if im is None:
                continue
            res = v.validate({"id": os.path.basename(fp), "rgb": im})
            b = res.rgb["brisque"] if res.rgb else float("nan")
            recs.append((os.path.basename(fp), res.verdict.value, b))
        bs = np.array([r[2] for r in recs])
        passed = sum(1 for r in recs if r[1] != "FAIL")
        print(f"\n  [{tag}] {len(recs)} 张")
        print(f"    BRISQUE mean={bs.mean():.2f} std={bs.std():.2f} "
              f"min={bs.min():.2f} max={bs.max():.2f}")
        print(f"    通过率(非FAIL): {passed}/{len(recs)} = "
              f"{100*passed/len(recs):.1f}%")
        return recs, bs

    real_files = _scan(frames_dir, n=n_real)
    gen_files = _scan(gen_dir)
    real_recs, real_bs = run(real_files, "真实帧")
    gen_recs, gen_bs = run(gen_files, "生成图")

    print("\n  [生成图逐张]")
    for name, verdict, b in gen_recs:
        print(f"    {name:22s} brisque={b:6.2f}  {verdict}")

    # 报告落盘
    report = {
        "threshold": thr,
        "real": {"n": len(real_recs),
                 "brisque_mean": float(real_bs.mean()),
                 "brisque_std": float(real_bs.std()),
                 "pass_rate": sum(1 for r in real_recs if r[1] != "FAIL") / len(real_recs)},
        "generated": {"n": len(gen_recs),
                      "brisque_mean": float(gen_bs.mean()),
                      "brisque_std": float(gen_bs.std()),
                      "records": [{"name": n_, "verdict": v_, "brisque": b_}
                                  for n_, v_, b_ in gen_recs]},
    }
    out = Path(__file__).parent.parent / "reports" / "e2e_test_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告写入 → {out.resolve()}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent  # 项目根
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",
                    default=str(base / "4-Transformer/output/gt/frames"))
    ap.add_argument("--gen",
                    default=str(base / "5-Controlnet/outputs"))
    ap.add_argument("--n-real", type=int, default=50)
    ap.add_argument("--skip-unit", action="store_true")
    args = ap.parse_args()

    if not args.skip_unit:
        unit_tests()
    integration_tests(args.frames, args.gen, args.n_real)
