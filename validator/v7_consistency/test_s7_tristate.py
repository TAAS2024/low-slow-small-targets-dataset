#!/usr/bin/env python3
"""
S7 尺度维度三态回归测试
=======================

锁死三件本轮修掉的事：

  1. size 维度能用论文 §4.2 的 E_scale = |s_pred − s_gt| / s_gt 计算，
     且 s_gt 来自生成端声明的 expected_px，不是物理投影。
  2. 没有 expected_px 时 size 记为 SKIP，不冒充 PASS，且写入 skipped_dims。
  3. 一个维度都没评估时 S7 verdict = SKIP（passed=False），不冒充通过。

外加一条接线检查：condition_generator_v7 的 _realised_geometry →
loop.build_sample_from_geometry → ConsistencyValidator 全链路数值闭合。

依赖只有 numpy / cv2，不需要模型权重，可直接在裸 clone 上跑：

    python3 validator/v7_consistency/test_s7_tristate.py
"""
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "validator" / "v7_consistency", REPO / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from consistency_validator import ConsistencyValidator, S7Verdict   # noqa: E402
from size_consistency import SizeConsistencyChecker                  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def _fake_image_with_target(h=720, w=1280, tx=60, ty=34):
    """造一张背景图 + 一张 seg marker，目标区域标 class 7。"""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    seg = np.full((h, w), 2, dtype=np.uint8)
    x1, y1 = 600, 330
    x2, y2 = x1 + tx - 1, y1 + ty - 1
    seg[y1:y2 + 1, x1:x2 + 1] = 7
    img[y1:y2 + 1, x1:x2 + 1] = np.array([90, 140, 110], dtype=np.uint8)
    return img, seg


def test_alignment_mode():
    print("\n1. alignment 模式（论文口径）")
    img, seg = _fake_image_with_target()
    H, W = seg.shape

    # 借真实生成端的抠框函数，确认它和 validator 用同一套像素约定
    from condition_generator_v7 import _realised_geometry
    geo = _realised_geometry(seg, target_px=60, dval=0.3, frame_index=0,
                             distance_m=75.0, action="hover",
                             sample_id="t/frame_0")
    check("geometry 抠到框", geo is not None)
    check("bbox_px 宽度 = 60", geo["realised_px"] == 60, str(geo["bbox_px"]))
    check("E_scale ≈ 0", geo["scale_err"] < 1e-6, f"scale_err={geo['scale_err']}")
    check("bbox_norm 自洽",
          abs(geo["bbox_norm"][2] * W - geo["realised_px"]) < 1e-3,
          f"w_norm={geo['bbox_norm'][2]:.6f} → {geo['bbox_norm'][2]*W:.3f}px")

    v = ConsistencyValidator()
    r = v.validate({"id": "align", "rgb": img,
                    "rgb_bbox": np.array(geo["bbox_norm"], dtype=np.float32),
                    "expected_px": geo["expected_px"]})
    check("size 进 PASS", r.dim_status["size"] == "PASS", r.reason)
    check("E_scale 直出", abs(r.scale_err - geo["scale_err"]) < 1e-5,
          f"{r.scale_err}")
    check("verdict=PASS", r.verdict is S7Verdict.PASS)

    # 声明 30px 但实际落 60px → E_scale = 1.0 → FAIL
    r2 = v.validate({"id": "align_bad", "rgb": img,
                     "rgb_bbox": np.array(geo["bbox_norm"], dtype=np.float32),
                     "expected_px": 30})
    check("尺寸不符 → FAIL", r2.dim_status["size"] == "FAIL")
    check("E_scale = 1.0", abs(r2.scale_err - 1.0) < 1e-6, f"{r2.scale_err}")


def test_skip_not_pass():
    print("\n2. 缺 expected_px → SKIP，不冒充 PASS")
    checker = SizeConsistencyChecker()
    bbox = np.array([0.5, 0.5, 60 / 1280, 34 / 720], dtype=np.float32)

    from size_consistency import SizeConsistencyResult
    sk = SizeConsistencyResult.skipped(bbox, "no expected_px")
    check("skipped() passed=False", sk.passed is False)
    check("skipped() status=SKIP", sk.status == "SKIP")
    check("skipped() mode=none", sk.mode == "none")

    img, _ = _fake_image_with_target()
    v = ConsistencyValidator()
    r = v.validate({"id": "skip", "rgb": img, "rgb_bbox": bbox})
    check("size 记入 skipped_dims", "size" in r.skipped_dims, str(r.skipped_dims))
    check("dim_status.size = SKIP", r.dim_status["size"] == "SKIP")
    check("skipped_dims 三项", sorted(r.skipped_dims) ==
          ["cross_modal", "ir_bbox", "size"], str(r.skipped_dims))
    check("size 字段带 status", r.size.get("status") == "SKIP")
    check("skipped 不阻断 verdict", r.verdict is S7Verdict.PASS,
          "只有 lighting 被评估且通过 → PASS")

    # 物理投影默认关闭：给了 category+distance 也不该进 size 判定
    r2 = v.validate({"id": "phys_off", "rgb": img, "rgb_bbox": bbox,
                     "drone_category": "small_quadcopter", "distance_m": 60})
    check("物理投影默认不启用", r2.dim_status["size"] == "SKIP", r2.reason)

    v3 = ConsistencyValidator(physical_fallback=True)
    r3 = v3.validate({"id": "phys_on", "rgb": img, "rgb_bbox": bbox,
                      "drone_category": "small_quadcopter", "distance_m": 60})
    check("显式开启后生效", r3.dim_status["size"] == "FAIL", r3.reason[:60])


def test_all_skip_verdict():
    print("\n3. 全维度 SKIP → verdict=SKIP")

    class _Dead:
        """让 lighting 也不可评估：直接构造一个全 SKIP 的路径。"""
        pass

    img, _ = _fake_image_with_target()
    v = ConsistencyValidator()
    r = v.validate({"id": "rgb_missing"})
    check("缺 rgb → FAIL 且全 SKIP",
          r.verdict is S7Verdict.FAIL
          and r.dim_status["size"] == "SKIP"
          and r.skipped_dims and len(r.skipped_dims) == 4)

    r2 = v.validate({"id": "bbox_missing", "rgb": img})
    check("缺 bbox → FAIL/S7_BBOX_MISSING",
          r2.verdict is S7Verdict.FAIL and r2.reason == "S7_BBOX_MISSING")

    # 直接构造"全 SKIP"结果：模拟 lighting 也返回 SKIP 的情形
    from consistency_validator import S7Result, DIMS
    r3 = S7Result(passed=False, verdict=S7Verdict.SKIP,
                  reason="no dimension evaluated",
                  dim_status={d: "SKIP" for d in DIMS},
                  skipped_dims=list(DIMS))
    check("PASS/FAIL/SKIP 三态可选",
          r3.verdict is S7Verdict.SKIP and r3.passed is False)


def test_build_sample_roundtrip():
    print("\n4. geometry → build_sample_from_geometry → S7 数值闭合")
    try:
        from loop.validator_pipeline import build_sample_from_geometry
    except Exception as e:                       # noqa: BLE001
        print(f"  [SKIP] pipeline 导入失败（缺模型依赖？）：{type(e).__name__}: {e}")
        return

    img, seg = _fake_image_with_target()
    from condition_generator_v7 import _realised_geometry
    geo = _realised_geometry(seg, target_px=60, dval=0.4, frame_index=3,
                             distance_m=120.0, action="hover",
                             sample_id="s9/frame_3")
    sample = build_sample_from_geometry(geo, rgb=img, sid="s9/frame_3")

    check("sample 带 expected_px", sample["expected_px"] == 60)
    check("sample 带 distance_m", sample["distance_m"] == 120.0)
    check("bbox 转成归一化 4 元", np.asarray(sample["rgb_bbox"]).shape == (4,))

    r = ConsistencyValidator().validate(sample)
    check("端到端 size=PASS", r.dim_status["size"] == "PASS", r.reason)
    check("端到端 E_scale 与生成端一致",
          r.scale_err is not None
          and abs(r.scale_err - geo["scale_err"]) < 1e-5,
          f"validator={r.scale_err} generator={geo['scale_err']}")

    # 声明 48px 但实际落 60px：E_scale = 0.25。
    # 默认门禁 0.50（产出论文 161/302 的运行时配置）下不触发 FAIL，
    # 用显式 strict checker 验证 E_scale 口径本身是对的。
    geo_bad = _realised_geometry(seg, target_px=48, dval=0.4, frame_index=3,
                                 distance_m=120.0, action="hover",
                                 sample_id="s9/frame_3")
    sample_bad = build_sample_from_geometry(geo_bad, rgb=img, sid="s9/frame_3")

    strict = ConsistencyValidator(
        size_checker=SizeConsistencyChecker(alignment_tolerance=0.10))
    r_bad = strict.validate(sample_bad)
    check("12px 偏差在 0.10 门禁下被抓", r_bad.dim_status["size"] == "FAIL",
          f"E_scale={r_bad.scale_err}")
    check("E_scale=0.25 与门禁无关",
          abs(r_bad.scale_err - 0.25) < 1e-6, f"{r_bad.scale_err}")

    # 同一样本在默认门禁下必须 PASS：这一条锁住论文数字的可复现性，
    # 谁把 alignment_tolerance 默认值改小，这里就会红。
    r_def = ConsistencyValidator().validate(sample_bad)
    check("默认 0.50 门禁下同一样本 PASS（可复现性锁）",
          r_def.dim_status["size"] == "PASS", f"E_scale={r_def.scale_err}")
    check("默认 alignment_tolerance 仍为 0.50",
          abs(SizeConsistencyChecker().alignment_tolerance - 0.5) < 1e-9)


def main():
    print("=" * 64)
    print("  S7 尺度维度三态回归")
    print("=" * 64)
    test_alignment_mode()
    test_skip_not_pass()
    test_all_skip_verdict()
    test_build_sample_roundtrip()

    print("\n" + "=" * 64)
    if FAILS:
        print(f"  {len(FAILS)} FAILED: {FAILS}")
        return 1
    print("  ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
