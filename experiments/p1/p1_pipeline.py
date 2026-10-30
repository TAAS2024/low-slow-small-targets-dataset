#!/usr/bin/env python3
"""
P1 实验管线 — 批量生成 + 验证 + 对齐精度评估
================================================
对齐任务清单 v4 战略：
  - 核心卖点：对齐精度评估（RGB↔Bbox / RGB↔IR / 三元组一致性）
  - 生成质量：BRISQUE（piq）
  - 验证：S6 质量 + S7 一致性（含 bbox 对齐）
  - 消融 E1：标记注入（ControlNet 重绘）vs 硬贴（skip_cnet）

用法：
  python p1_pipeline.py --mode single          # 单帧端到端验证
  python p1_pipeline.py --mode batch --num 30  # 批量生成
  python p1_pipeline.py --mode ablation_e1     # 消融 E1
"""

import os, sys, json, time, warnings, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# p1_pipeline.py 位于 <项目根>/experiments/p1/，上溯三级到项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 路径注入 ──────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "3-LLM starter"))
sys.path.insert(0, str(PROJECT_ROOT / "5-Controlnet"))
sys.path.insert(0, str(PROJECT_ROOT / "2-Lora training"))
S6_DIR = PROJECT_ROOT / "6-Validator" / "S6 RGB 图像质量检查" / "code"
S7_DIR = PROJECT_ROOT / "6-Validator" / "S7 无人机与场景一致性检查" / "code"
for _d in (S6_DIR, S7_DIR):
    sys.path.insert(0, str(_d))

from condition_generator_v7 import extract_depth_matched, heuristic_segment
from rgb2ir_converter import rgb_to_whitehot
from lora_inpainter_v7 import LoraInpainterV7
from quality_validator import QualityValidator
from consistency_validator import ConsistencyValidator

BG_ROOT = PROJECT_ROOT / "1-background-pool" / "curated_backgrounds"
OUT_ROOT = PROJECT_ROOT / "experiments" / "p1"
SCENES = ["pure_sky", "coastal", "urban", "rural", "forest",
          "mountain", "desert", "industrial", "airfield"]


# ═══════════════════════════════════════════════════════════
# 几何工具
# ═══════════════════════════════════════════════════════════

def mask_to_bbox(mask: np.ndarray, thr: int = 64) -> np.ndarray:
    """二值 mask → [x1,y1,x2,y2] 像素 bbox"""
    ys, xs = np.where(mask > thr)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_cxcywh(bbox, W, H):
    """[x1,y1,x2,y2] → 归一化 [cx,cy,w,h]"""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                     (x2 - x1) / W, (y2 - y1) / H], dtype=np.float32)


def bbox_iou(a, b):
    """两个 [x1,y1,x2,y2] 的 IoU"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def mask_contrast(img_rgb, mask, ring=6):
    """mask 区域内像素 vs 外环像素的平均绝对差（可辨识度）"""
    m = mask > 64
    if m.sum() == 0:
        return 0.0
    k = np.ones((ring, ring), np.uint8)
    dil = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    ring_mask = dil & (~m)
    if ring_mask.sum() == 0:
        return 0.0
    inner = img_rgb[m].astype(np.float32).mean()
    outer = img_rgb[ring_mask].astype(np.float32).mean()
    return float(abs(inner - outer))


def edge_artifact_score(final_rgb, composite_rgb, mask, ring=4):
    """mask 边界处的 diff 梯度（边缘伪影率，越低融合越自然）"""
    m = mask > 64
    if m.sum() == 0:
        return 0.0
    k = np.ones((ring, ring), np.uint8)
    dil = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    ero = cv2.erode(m.astype(np.uint8), k).astype(bool)
    boundary = dil & (~ero)
    if boundary.sum() == 0:
        return 0.0
    diff = cv2.absdiff(final_rgb, composite_rgb).astype(np.float32)
    return float(diff[boundary].mean())


# ═══════════════════════════════════════════════════════════
# P1 实验主类
# ═══════════════════════════════════════════════════════════

class P1Experiment:
    def __init__(self, out_dir=OUT_ROOT):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.inpainter = None
        self.s6 = None
        self.s7 = None

    def load(self):
        t0 = time.time()
        print("加载 LoraInpainterV7 (SD1.5 + ControlNet depth/seg + LoRA) ...")
        self.inpainter = LoraInpainterV7()
        self.inpainter.load()
        print(f"  inpainter 加载完成 ({time.time()-t0:.1f}s)")
        self.s6 = QualityValidator()
        self.s7 = ConsistencyValidator()
        print("  S6/S7 Validator 就绪")

    def unload(self):
        if self.inpainter:
            self.inpainter.unload()
            self.inpainter = None
        torch = sys.modules.get("torch")
        if torch:
            torch.cuda.empty_cache()

    # ── 单帧处理 ─────────────────────────────────────────
    def process_frame(self, bg_path, nu, nv, target_px, time_of_day="afternoon",
                      skip_cnet=False, sample_id="s0", save=True):
        """生成一帧并返回完整评估结果"""
        # 1. 读背景 + 提取 depth/seg
        original = Image.open(bg_path).convert("RGB")
        bg_arr = np.array(original)
        bg_bgr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2BGR)
        H, W = bg_arr.shape[:2]
        depth = extract_depth_matched(Image.fromarray(bg_arr), (W, H))
        seg = heuristic_segment(bg_arr)

        # 2. 生成（标记注入 或 硬贴）
        # dark 场景（dusk/night）增强：低照度下 depth/seg 条件信噪比下降，
        # 提高重绘强度与步数以抑制边缘伪影（P1 观测到 edge_artifact 偏高）
        is_dark = time_of_day in ("dusk", "night")
        blend_steps = 25 if is_dark else 15
        blend_strength = 0.75 if is_dark else 0.65
        out = self.inpainter.inpaint(
            background_bgr=bg_bgr, bg_depth=depth, bg_seg=seg,
            nu=nu, nv=nv, target_px=target_px, dval=0.4, class_id=7,
            gen_seed=int(time.time_ns() % (2**31)), gen_steps=20,
            blend_seed=int(time.time_ns() % (2**31)) + 100000,
            blend_steps=blend_steps, blend_strength=blend_strength, guidance=7.5,
            time_of_day=time_of_day, weather="clear",
            use_lora=True, skip_cnet=skip_cnet,
        )
        final_bgr = out["result"]
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

        # 3. GT bbox（drone_mask = 标记注入位置 = 系统承诺位置）
        gt_bbox = mask_to_bbox(out.get("drone_mask"))
        # 期望位置/尺寸（系统参数）
        exp_cx, exp_cy, exp_px = out.get("cx"), out.get("cy"), out.get("actual_px")
        exp_bbox = np.array([exp_cx - exp_px/2, exp_cy - exp_px/2,
                             exp_cx + exp_px/2, exp_cy + exp_px/2], dtype=np.float32)

        # 4. IR 转换（白热，gain=1.5 补偿低照度场景对比度衰减）
        ir_rgb = rgb_to_whitehot(final_bgr, gain=1.5)  # BGR in → RGB out

        # 5. 对齐精度（RGB↔Bbox：GT vs 期望）
        align = {}
        if gt_bbox is not None:
            gt_cx, gt_cy = (gt_bbox[0]+gt_bbox[2])/2, (gt_bbox[1]+gt_bbox[3])/2
            gt_w, gt_h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
            align["pos_err_px"] = float(np.hypot(gt_cx-exp_cx, gt_cy-exp_cy))
            align["scale_err"] = float(abs(max(gt_w, gt_h) - exp_px) / exp_px)
            align["iou"] = float(bbox_iou(gt_bbox, exp_bbox))
        else:
            align["pos_err_px"] = align["scale_err"] = align["iou"] = None

        # RGB↔IR 对齐（确定性转换：mask 内对比度 = 可辨识度）
        mask = out.get("drone_mask")
        align["ir_contrast"] = mask_contrast(ir_rgb, mask) if mask is not None else None

        # 6. S6 质量验证
        s6_res = self.s6.validate({
            "id": sample_id, "rgb": final_rgb, "ir": ir_rgb})
        # 7. S7 一致性验证（含 bbox 对齐）
        s7_res = self.s7.validate({
            "id": sample_id, "rgb": final_rgb, "ir": ir_rgb,
            "rgb_bbox": bbox_cxcywh(gt_bbox, W, H) if gt_bbox is not None else None,
            "ir_bbox": bbox_cxcywh(gt_bbox, W, H) if gt_bbox is not None else None,
            "expected_px": target_px,
            "drone_category": "pocket_uav",
            "distance_m": round(3000 / max(target_px, 12), 1),
        })

        # 8. 边缘伪影（融合质量）
        comp = out.get("composite")
        if comp is not None and mask is not None:
            edge = edge_artifact_score(final_rgb, comp, mask)
        else:
            edge = None

        result = {
            "id": sample_id, "bg": str(Path(bg_path).name),
            "nu": nu, "nv": nv, "target_px": target_px, "time_of_day": time_of_day,
            "skip_cnet": skip_cnet,
            "W": W, "H": H,
            "gt_bbox": gt_bbox.tolist() if gt_bbox is not None else None,
            "align": align,
            "s6_passed": s6_res.passed, "s6_reason": s6_res.reason,
            "s6_brisque": getattr(s6_res, "rgb", {}).get("brisque", None) if hasattr(s6_res, "rgb") else None,
            "s7_passed": s7_res.passed, "s7_reason": s7_res.reason,
            "edge_artifact": edge,
        }
        # 从 S6 result 提取 brisque（dataclass asdict 结构）
        try:
            from dataclasses import asdict
            d6 = asdict(s6_res)
            result["s6_brisque"] = d6.get("rgb", {}).get("brisque")
            result["s6_detail"] = d6
        except Exception:
            pass
        try:
            from dataclasses import asdict
            result["s7_detail"] = asdict(s7_res)
        except Exception:
            pass

        # 9. 保存产物
        if save:
            fid = sample_id
            cv2.imwrite(str(self.out_dir / f"{fid}_rgb.png"), final_bgr)
            cv2.imwrite(str(self.out_dir / f"{fid}_ir.png"),
                        cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR))
            if gt_bbox is not None:
                viz = final_bgr.copy()
                cv2.rectangle(viz, (int(gt_bbox[0]), int(gt_bbox[1])),
                              (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)
                cv2.imwrite(str(self.out_dir / f"{fid}_bbox.png"), viz)
            if comp is not None:
                cv2.imwrite(str(self.out_dir / f"{fid}_composite.png"), comp)
        return result

    # ── 批量 ────────────────────────────────────────────
    def run_batch(self, num, skip_cnet=False, tag=""):
        results = []
        bg_pool = []
        for sc in SCENES:
            d = BG_ROOT / sc
            if d.exists():
                bg_pool += sorted(d.glob("bg_*.jpg"))
        bg_pool = [p for p in bg_pool if p.exists()]
        if not bg_pool:
            print("!! 背景池为空"); return results
        print(f"背景池: {len(bg_pool)} 张")

        rng = np.random.default_rng(42)
        for i in range(num):
            bg = bg_pool[rng.integers(len(bg_pool))]
            nu = float(rng.uniform(0.2, 0.8))
            nv = float(rng.uniform(0.15, 0.6))
            px = int(rng.integers(30, 90))
            tod = rng.choice(["afternoon", "dusk", "night"])
            sid = f"{tag}{i:03d}"
            print(f"\n[{i+1}/{num}] {sid} bg={bg.name} nu={nu:.2f} nv={nv:.2f} px={px} tod={tod}")
            r = self.process_frame(bg, nu, nv, px, tod, skip_cnet=skip_cnet,
                                   sample_id=sid)
            results.append(r)
            # 定期 flush
            if (i+1) % 5 == 0:
                self._dump(results, f"partial_{tag}.json")
        return results

    def _dump(self, results, fname):
        path = self.out_dir / fname
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  → 已写 {path}")


def summarize(results):
    """汇总统计"""
    n = len(results)
    if n == 0:
        return {}
    pos = [r["align"]["pos_err_px"] for r in results if r["align"]["pos_err_px"] is not None]
    scale = [r["align"]["scale_err"] for r in results if r["align"]["scale_err"] is not None]
    iou = [r["align"]["iou"] for r in results if r["align"]["iou"] is not None]
    ir = [r["align"]["ir_contrast"] for r in results if r["align"]["ir_contrast"] is not None]
    brisque = [r.get("s6_brisque") for r in results if r.get("s6_brisque") is not None]
    edge = [r["edge_artifact"] for r in results if r["edge_artifact"] is not None]
    s6_pass = sum(1 for r in results if r["s6_passed"])
    s7_pass = sum(1 for r in results if r["s7_passed"])

    def stat(x):
        if not x:
            return {"mean": None, "std": None, "min": None, "max": None}
        a = np.array(x, dtype=np.float32)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "n": n,
        "s6_pass_rate": s6_pass / n, "s7_pass_rate": s7_pass / n,
        "align_pos_err_px": stat(pos),
        "align_scale_err": stat(scale),
        "align_iou": stat(iou),
        "ir_contrast": stat(ir),
        "brisque": stat(brisque),
        "edge_artifact": stat(edge),
    }


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single", choices=["single", "batch", "ablation_e1"])
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    exp = P1Experiment(out_dir=args.out)
    exp.load()

    try:
        if args.mode == "single":
            # 单帧端到端
            bg = BG_ROOT / "pure_sky" / "bg_000001.jpg"
            if not bg.exists():
                bg = next((BG_ROOT / "pure_sky").glob("*.jpg"))
            print(f"单帧测试: {bg}")
            r = exp.process_frame(bg, 0.5, 0.35, 60, "afternoon", sample_id="single")
            print(json.dumps(r, indent=2, ensure_ascii=False))
            print("\n=== 汇总 ===")
            print(json.dumps(summarize([r]), indent=2, ensure_ascii=False))
            exp._dump([r], "single_result.json")

        elif args.mode == "batch":
            res = exp.run_batch(args.num, skip_cnet=False, tag="b")
            exp._dump(res, "batch_results.json")
            print("\n=== 批量汇总 ===")
            print(json.dumps(summarize(res), indent=2, ensure_ascii=False))
            (exp.out_dir / "batch_summary.json").write_text(
                json.dumps(summarize(res), indent=2, ensure_ascii=False))

        elif args.mode == "ablation_e1":
            # 消融 E1：标记注入 vs 硬贴（同背景同参数对比）
            bg = BG_ROOT / "coastal" / "bg_000001.jpg"
            if not bg.exists():
                bg = next((BG_ROOT / "coastal").glob("*.jpg"))
            pairs = []
            for i in range(args.num):
                nu, nv, px = 0.5, 0.4, 60
                print(f"\n=== 消融对 {i} 标记注入 ===")
                r_inject = exp.process_frame(bg, nu, nv, px, "afternoon",
                                             skip_cnet=False, sample_id=f"inj{i}")
                print(f"=== 消融对 {i} 硬贴 ===")
                r_paste = exp.process_frame(bg, nu, nv, px, "afternoon",
                                            skip_cnet=True, sample_id=f"paste{i}")
                pairs.append({"inject": r_inject, "paste": r_paste})
            exp._dump(pairs, "ablation_e1.json")
            inj = summarize([p["inject"] for p in pairs])
            pas = summarize([p["paste"] for p in pairs])
            print("\n=== 消融 E1 汇总 ===")
            print("标记注入:", json.dumps(inj, indent=2, ensure_ascii=False))
            print("硬贴:    ", json.dumps(pas, indent=2, ensure_ascii=False))
            (exp.out_dir / "ablation_e1_summary.json").write_text(
                json.dumps({"inject": inj, "paste": pas}, indent=2, ensure_ascii=False))
    finally:
        exp.unload()


if __name__ == "__main__":
    main()
