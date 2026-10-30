#!/usr/bin/env python3
"""make_paper_v9.py — 把 0-works-v8.docx 更新为 v9（口径 + 数据 + 图例）
=============================================================================
两条独立的编辑线：
  A) 静态口径修正（STATIC_EDITS）：与实验数据无关，描述与代码实现对齐。
  B) 数据驱动修正（NUM_ANCHORS / TABLE_EDITS）：锚点替换，旧串来自 v8 原文，
     新值实时读取 experiments/bcd/out_* 的评测 JSON。
     若某段数据尚未生成 → 该锚点跳过并记入清单（PENDING），不写错数字。

幂等：每次都从 v8 重新派生 v9，可反复运行。v8 永不被改写。

用法:
    python3 make_paper_v9.py             # 应用全部（缺数据的锚点留 PENDING）
    python3 make_paper_v9.py --report    # 只出清单，不写 docx

输出:
    ·重点节点总结笔记/论文写作/0-works-v9.docx
    ·重点节点总结笔记/论文写作/0-works-v9-变更说明.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
PAPER = PROJ / "·重点节点总结笔记" / "论文写作"
SRC = PAPER / "0-works-v8.docx"
DST = PAPER / "0-works-v9.docx"
LOG = PAPER / "0-works-v9-变更说明.md"
FIGDIR = PAPER / "figures"

# ── 段落索引（v8 实测；脚本开头会断言这些索引的锚点仍在）──
P_ABS = 3
P_CONTRIB2 = 20
P_CONTRIB5 = 21
P_M5_BODY = 74
P_M5_CFG = 76
P_M5_BLEND = 78
P_M5_WHERE = 80
P_FRRM = 82
P_SEC41_A = 125
P_SEC41_B = 126
P_SEC43_A = 143
P_SEC43_B = 143   # v8 [143] 才是 300-pair 正文段；旧值 147 错位（指向昼夜不变性段）
P_SEC43_TOD = 147
P_SEC43_C = 148
P_E1_INTRO = 167
P_E1_STAT = 170
P_E1_LCR = 171
P_E1_PROXY = 172
P_E3 = 177
P_E4 = 179
P_E5 = 182
P_E6 = 184
P_B_INTRO = 186
P_B_CAVEAT = 188
P_PROXY = 190
P_SEC49 = 192          # v8 [192] = §4.5.9「Within the E1 pairing...」段（v9 显示为 [194]）
P_CONCL_A = 209
P_CONCL_B = 210
P_CONCL_C = 211
# Implementation revision 披露段插入到 §4.2 之前（源文档第 127 段前）
P_DISCLOSE_AT = 127

# ⑪ Figure 7（消融 E1 定性对比）插入到 §4.5.1 讨论段之前（v8 第 170 段前）
P_FIG7_PROBE = "The two arms are statistically indistinguishable"
P_FIG7_BEFORE = 170
FIG7_PATH = FIGDIR / "fig7_ablation.png"
FIG7_WIDTH_IN = 4.8
S_FIG7_CAP = ("Ablation E1 at matched conditioning points: hard-paste baseline (left) versus "
              "marker injection with depth-conditioned ControlNet repainting (right). Each row "
              "reuses the same conditioning point in both arms; the ground-truth bounding box "
              "is overlaid in green.")
S_FIG7_REF = " Figure 7 shows the two arms side by side at matched conditioning points."


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════
def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_any(p: Path):
    """json 可能是 list，也可能是 dict；失败返回 None。"""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def dig(obj, path, default=None):
    """obj 里按 'a.b.c' 取值；任一层缺失返回 default。"""
    cur = obj
    for k in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(k)]
                continue
            except Exception:
                return default
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_find(obj, key, default=None):
    """递归找第一个同名 key（用于兼容未知 JSON 结构）。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = deep_find(v, key, None)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = deep_find(v, key, None)
            if r is not None:
                return r
    return default


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mcnemar_exact(b: int, c: int):
    """配对二值检验（McNemar 精确二项，双尾）。

    b / c 是两个方向的不一致对数。返回 (p, n_discordant)。
    用于判断某个消融臂与对照臂在「检出率」这类二值下游指标上是否可区分 —
    检出力极低（背景假阳性淹没目标贡献），因此论文 §4.5.8 的结论是
    「不可区分」而非「反向 trade-off」。见 make_paper_v9.py 的 e1_proxy /
    proxy_block / sec49_block 锚点。
    """
    n = b + c
    if n == 0:
        return None, 0
    k = min(b, c)
    # 双尾精确二项，p = 0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail), n


def wilcoxon_signed_rank(diffs):
    """配对 Wilcoxon 符号秩检验（正态近似 + 并列校正），双尾。

    用于连续型下游指标（best-box IoU）：与二值检出率相比，同一批样本下
    功效高得多 —— E4 的「双条件 vs 仅深度」在 IoU 上 p = 0.0017 显著，
    而在 R_aerial 上 p = 0.12 不显著。
    """
    nz = [x for x in diffs if x is not None and abs(x) > 1e-12]
    n = len(nz)
    if n == 0:
        return None, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
            j += 1
        avg = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if nz[i] > 0)
    wm = sum(ranks[i] for i in range(n) if nz[i] < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    from collections import Counter
    tie = Counter([round(abs(x), 9) for x in nz])
    tv = sum(t ** 3 - t for t in tie.values())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tv / 48.0
    if var <= 0:
        return None, n
    z = (w - mu + 0.5 * math.copysign(1, mu - w)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, n


# 每个臂在 out_downstream/downstream.json 里应有的样本数。
# 守卫用途：ablation_driver.gen() 会用 flush_every=1 逐条覆写 manifest.json，
# 导致 eval_downstream.py 只读到 1 个样本、产出 n=1 的伪指标。2026-10-06 曾发生，
# 且本脚本无校验，把伪指标静默写进了 0-works-v9.docx。n 不符预期 → 拒发该臂数字。
EXPECT_DS_N = {
    "out_D": 300,
    "out_E1_full": 150, "out_E1_paste": 150,
    "out_E3_lora": 100, "out_E3_nolora": 100,
    "out_E4_both": 100, "out_E4_depth": 100, "out_E4_seg": 100,
}


class M:
    """全部论文数值的唯一来源。缺文件时对应属性为 None，锚点自动跳过。"""

    def __init__(self, bcd: Path):
        self.b = bcd
        self._d_sum = _load(bcd / "out_D_eval" / "summary.json")
        self._d_res = _load(bcd / "out_D_eval" / "results.json")
        self._e1 = {k: _load(bcd / f"out_E1_{k}" / "metrics.json")
                    for k in ("full", "paste")}
        self._e1f = _load(bcd / "out_E1_fusion" / "fusion.json")
        self._e3 = {k: _load(bcd / f"out_E3_{k}" / "metrics.json")
                    for k in ("lora", "nolora")}
        self._e4 = {k: _load(bcd / f"out_E4_{k}" / "metrics.json")
                    for k in ("both", "depth", "seg")}
        self._e4f = _load(bcd / "out_fusion_e1e4" / "fusion.json")
        self._e5 = {k: _load(bcd / f"out_E5_{k}" / "metrics.json")
                    for k in ("fixed", "indep")}
        self._e6 = {k: _load(bcd / f"out_E6_N{k}" / "metrics.json")
                    for k in ("1", "2", "3")}
        self._ds = _load(bcd / "out_downstream" / "downstream.json")
        self._ds_bad = set()   # 已被守卫拒发的臂，只警告一次
        self._tim = _load_any(bcd / "out_D" / "timing.json")
        self._plan = _load_any(bcd / "out_D" / "plan.json")
        self._b = {k: _load(bcd / f"out_B_{k}" / "curve.json") for k in ("A1", "A2", "A3")}

    # ── 生成耗时（out_D/timing.json × plan.json）──
    def _lat_ok(self):
        t = self._tim
        return [float(x["sec"]) for x in t if x.get("ok")] if t else []

    def lat(self):
        v = self._lat_ok()
        if not v:
            return None
        return {"mean": mean(v), "median": median(v), "min": min(v), "max": max(v), "n": len(v)}

    def lat_tod(self):
        t, pl = self._tim, self._plan
        if not t or not pl:
            return {}
        tot = {}
        if isinstance(pl, dict):
            pl = pl.get("samples") or pl.get("plan") or []
        for s in pl:
            tot[s.get("sample_id")] = s.get("time_of_day")
        out = {}
        for x in t:
            if not x.get("ok"):
                continue
            d = tot.get(x.get("sample_id"))
            if d:
                out.setdefault(d, []).append(float(x["sec"]))
        return {k: mean(v) for k, v in out.items()}

    # ── B 组闭环 ──
    def bcy(self, arm):
        j = self._b.get(arm)
        return dig(j, "results") if j else None

    def bcy_rate(self, arm):
        r = self.bcy(arm)
        return [x.get("acceptance_over_targets") for x in r] if r else []

    def bcy_tot(self, arm):
        r = self.bcy(arm)
        if not r:
            return (None, None)
        return (sum(x.get("attempts", 0) for x in r),
                sum(x.get("n_accepted", 0) for x in r))

    # ── 便捷：D 组汇总 ──
    @property
    def d(self):
        s = self._d_sum
        return dig(s, "summary", s) if s else None

    def dp(self, *path):
        return dig(self.d, ".".join(path))

    @property
    def d_samples(self):
        return dig(self._d_res, "samples") if self._d_res else None

    # ── 派生量：逐时段 / 逐场景 / 大小目标分层 / 画布对角 ──
    def tod_stat(self, key):
        """key in {pos_err_px, scale_err, iou} → {tod: mean}"""
        ss = self.d_samples
        if not ss:
            return {}
        out = {}
        for s in ss:
            t, v = s.get("time_of_day"), dig(s, f"alignment.{key}")
            if t is None or v is None:
                continue
            out.setdefault(t, []).append(float(v))
        return {k: mean(v) for k, v in out.items()}

    def scene_brisque(self):
        ss = self.d_samples
        if not ss:
            return {}
        out = {}
        for s in ss:
            sc, b = s.get("scene"), s.get("brisque")
            if sc and b is not None:
                out.setdefault(sc, []).append(float(b))
        return {k: mean(v) for k, v in out.items()}

    def split_by_side(self, thr=40.0):
        """按 GT bbox 长边 < thr 切分 → (small, large) 样本列表"""
        ss = self.d_samples
        if not ss:
            return [], []
        small, big = [], []
        for s in ss:
            box = dig(s, "gt_bbox") or dig(s, "alignment.bbox")
            if not box or len(box) < 4:
                continue
            side = max(abs(box[2] - box[0]), abs(box[3] - box[1]))
            (small if side < thr else big).append(s)
        return small, big

    @staticmethod
    def agg(ss, key):
        return mean([dig(s, f"alignment.{key}") for s in ss])

    def median_diag(self):
        ss = self.d_samples
        if not ss:
            return None
        return median([math.hypot(s.get("W", 0), s.get("H", 0)) for s in ss])

    def d_median(self, key):
        ss = self.d_samples
        return median([s.get(key) for s in ss]) if ss else None

    def d_rate(self, key):
        """samples 里布尔字段的通过率"""
        ss = self.d_samples
        return mean([1.0 if s.get(key) else 0.0 for s in ss]) if ss else None

    def tod_brisque(self):
        ss = self.d_samples
        if not ss:
            return {}
        out = {}
        for s in ss:
            t, b = s.get("time_of_day"), s.get("brisque")
            if t and b is not None:
                out.setdefault(t, []).append(float(b))
        return {k: mean(v) for k, v in out.items()}

    # ── 消融组访问器 ──
    def e1(self, k):
        return self._e1.get(k)

    def e3(self, k):
        return self._e3.get(k)

    def e4(self, k):
        return self._e4.get(k)

    def e5(self, k):
        return self._e5.get(k)

    def e6(self, k):
        return self._e6.get(k)

    def ds(self, dirname, key, default=None):
        if not self._ds:
            return default
        exp = EXPECT_DS_N.get(dirname)
        if exp is not None:
            got = dig(self._ds, f"{dirname}.n")
            if got != exp:
                if dirname not in self._ds_bad:
                    self._ds_bad.add(dirname)
                    print(f"[WARN] downstream n mismatch for {dirname}: got {got!r}, "
                          f"expected {exp} — 拒发该臂下游指标"
                          f"（manifest 可能退化，先跑 repair_manifests.py --dry-run）",
                          file=sys.stderr)
                return default
        return dig(self._ds, f"{dirname}.{key}", default)

    def ds_rows(self, dirname):
        """逐样本行（含 sample_id / aerial_hit / any_hit / best_iou）。"""
        if not self._ds:
            return {}
        rows = dig(self._ds, f"{dirname}.rows") or []
        return {r.get("sample_id"): r for r in rows if r.get("sample_id") is not None}

    def ds_pair(self, dir_a, dir_b):
        """两个下游臂的配对统计。

        返回 dict：
            n            共同样本数
            delta        检出率之差 R_aerial(a) - R_aerial(b)
            p            McNemar 精确二项双尾 p（二值检出率）
            b, c         两个方向的不一致对数
            diou         平均 best-box IoU 之差
            p_iou        配对 Wilcoxon 双尾 p（连续定位质量）
        """
        A, B = self.ds_rows(dir_a), self.ds_rows(dir_b)
        if not A or not B:
            return None
        common = sorted(set(A) & set(B))
        if not common:
            return None
        ah = [1 if A[i].get("aerial_hit") else 0 for i in common]
        bh = [1 if B[i].get("aerial_hit") else 0 for i in common]
        b = sum(1 for i in range(len(common)) if ah[i] and not bh[i])
        c = sum(1 for i in range(len(common)) if bh[i] and not ah[i])
        p, _ = mcnemar_exact(b, c)
        delta = (sum(ah) - sum(bh)) / len(common)
        aa = [A[i].get("best_iou") or 0.0 for i in common]
        bb = [B[i].get("best_iou") or 0.0 for i in common]
        diou = mean([aa[i] - bb[i] for i in range(len(common))])
        p_iou, _ = wilcoxon_signed_rank([aa[i] - bb[i] for i in range(len(common))])
        return {"n": len(common), "delta": delta, "p": p, "b": b, "c": c,
                "diou": diou, "p_iou": p_iou}

    def ds_p(self, dir_a, dir_b):
        s = self.ds_pair(dir_a, dir_b)
        return s["p"] if s else None

    def ds_delta(self, dir_a, dir_b):
        s = self.ds_pair(dir_a, dir_b)
        return s["delta"] if s else None

    # ── 融合指标（LCR / RLJ / BGE 及配对 Wilcoxon）──
    def _fus_json(self, which="E1"):
        return self._e1f if which == "E1" else self._e4f

    def fus(self, dirname, metric, field="mean", which="E1"):
        j = self._fus_json(which)
        if not j:
            return None
        for r in dig(j, "results") or []:
            if r.get("dir") == dirname:
                return dig(r, f"metrics.{metric}.{field}")
        return None

    def fus_p(self, metric, field="wilcoxon_p", which="E1", pair=None):
        j = self._fus_json(which)
        pr = dig(j, "pairs") if j else None
        if not pr:
            return None
        for e in pr:
            nm = str(e.get("pair"))
            if pair is None or (pair[0] in nm and pair[1] in nm):
                v = dig(e, f"metrics.{metric}.{field}")
                if v is not None:
                    return v
        return None


# ═══════════════════════════════════════════════════════════════════════
# A) 静态口径修正 —— 与实验数据无关，使描述与代码实现一致
# ═══════════════════════════════════════════════════════════════════════
S_M5_BODY = (
    "Module M5 performs localized target repainting into the host background using a "
    "dual-conditioned ControlNet [47] attached to a Stable Diffusion 1.5 image-to-image "
    "backbone. The repainting process must satisfy structural edge coherence, metric "
    "perspective, and undisturbed background conservation. The module conditions on the two "
    "spatial maps derived by the marker injection strategy of Section 3.3.3, so that the "
    "conditioning vector concatenates the depth-conditioned map D' with the "
    "segmentation-conditioned map S':"
)

S_M5_CFG = (
    "Denoising diffusion sampling follows classifier-free guidance parameterized by weighting "
    "scalar \u03c9, and partial denoising is governed by a strength parameter s. The sampler is "
    "initialised from the forward-diffused latent of the composite at step "
    "T\u2080 = \u230aN\u00b7s\u230b rather than from pure noise, so the number of denoising steps "
    "actually executed is \u230aN\u00b7s\u230b; the host background enters the reverse process as "
    "the image-to-image starting state instead of being regenerated:"
)

S_M5_BLEND = (
    "To keep the host background outside the insertion region unchanged, the repainted crop is "
    "recomposited with the pre-repaint composite through a feathered insertion mask:"
)

S_M5_WHERE = (
    "where M is the feathered insertion mask derived from the injected target silhouette, "
    "I_repaint is the ControlNet output and I_comp the pre-repaint composite. The feather band "
    "adapts to the target size, w = max(2 px, min(12 px, 0.05\u00b7e)) for a target edge length e, "
    "and is blurred with a Gaussian kernel of \u03c3 = max(0.8, w/2.5). Because the repainted "
    "crop retains residual high-frequency texture of its own, a boundary-harmonisation "
    "correction is applied outside the silhouette before the blend: the region eligible for "
    "replacement is extended to every pixel outside the silhouette whose mask weight exceeds "
    "0.002, and within that region the low-frequency component of the repainted crop is kept "
    "while its high-frequency component is taken from the pre-repaint composite, after which "
    "the region's per-channel mean and standard deviation are aligned to the composite. Pixels "
    "inside the silhouette are never modified, so the correction is a fidelity correction to "
    "the host background rather than a change to the synthesized target. Repainting is executed on a "
    "local crop that encloses the target with 2.5\u00d7 margin (at least 64 px per side); the crop is "
    "resampled to a 512\u00d7512 canvas, the native resolution of the SD 1.5 backbone, repainted "
    "there, and resampled back, so the full canvas is never re-rendered. Ground-truth "
    "bounding-box coordinates are extracted directly from the spatial extent of the injected "
    "silhouette."
)

S_FRRM = (
    "To reduce the computational overhead of generating multiple spatial permutations over an "
    "identical scene-target pair, the Focus Rapid Repositioning Mechanism (FRRM) applies a "
    "localized, cache-reusing rendering strategy. After an initial baseline generation has "
    "established the scene state, subsequent target repositioning operations reuse it directly: "
    "the background depth and segmentation maps (M3), the matted target layer and its silhouette "
    "(M4) and the sampling parameters of the base frame are carried over, so that only the "
    "composite, the marker injection and the ControlNet repaint of Section 3.4.3 are re-executed. "
    "Repainting is restricted to a local crop around the updated insertion coordinates, "
    "resampled to a 512\u00d7512 canvas, and the resulting patch is blended back with the feathered "
    "mask of Eq. (7). Because the target layer is reused rather than resampled, intra-sequence "
    "identity is preserved by construction while the insertion geometry remains free to vary over "
    "the trajectory. This localized execution removes the LoRA generation, silhouette matting and "
    "background-analysis stages from every frame after the first, reducing per-frame latency while "
    "maintaining boundary visual fidelity."
)

S_41_FIX_HEAD = ("Implementation revision. The repainting configuration reported here differs "
                 "from the one used in an earlier internal build, and the difference matters for "
                 "interpretation. In that earlier build the repainting module instantiated the "
                 "text-to-image ControlNet pipeline while supplying image-to-image arguments: the "
                 "keyword arguments carrying the conditioning maps and the denoising strength "
                 "were absorbed silently by the pipeline's catch-all parameter dict, so the depth "
                 "and segmentation conditions were never consumed, the composite itself was "
                 "passed to ControlNet as its single condition, and every call re-rendered the "
                 "region from pure noise with the strength parameter inert. An internal audit "
                 "isolated the fault by showing that outputs were bit-identical across five "
                 "different strength settings.")

S_41_FIX_TAIL = ("All quantitative results reported in Sections 4.3 to 4.5 were produced after "
                 "the module was corrected to the image-to-image ControlNet configuration "
                 "described in Section 3.4.3 - both spatial conditions active at a conditioning "
                 "scale of 0.85, partial denoising with s = 0.65 over N = 15 sampling steps, i.e. "
                 "9 executed denoising steps - and every experiment driver was re-run end to end "
                 "against the corrected build. The pre-correction artefacts are retained in the "
                 "project repository for provenance only and are not the source of any figure, "
                 "table or number in this paper. Because the corrected configuration consumes the "
                 "conditioning maps for the first time, the ablations of Section 4.5 are also the "
                                                                          "first measurement of what those maps contribute.")


# ═══════════════════════════════════════════════════════════════════════
# 诚信修正（2026-10-11）：把正文与代码/数据的真实行为对齐
#   ② M2 检索机制：v8 声称 CLIP 多准则排序，实际实现是「场景目录硬过滤 +
#      时段/天气标签分层匹配 + 层内均匀随机 + 滑动排除队列」。
#      见 0-workspace/app/background_searcher.py（BG_MATCH 默认 labels）。
#   ② 评测 harness 连标签匹配都不走：experiments/bcd/harness.py::plan()
#      在类内均匀随机取背景，tod/weather 由均衡循环赋值（是「请求的条件
#      标签」，不是背景图自身的元数据）。
#   ③ 背景池元数据覆盖率：37,852 张里 37,559 张有记录（差 293，mainly
#      mountain 304 / industrial 多 11）。
# ═══════════════════════════════════════════════════════════════════════

S_M2_OLD = ("Background retrieval in Module M2 executes multi-criteria ranking over indexed "
            "metadata vectors incorporating dominant chromatic histograms, estimated depth "
            "profiles, and visual embeddings extracted via a CLIP ViT-L/14 encoder. Scene "
            "category acts as a hard filter, while cosine similarity across CLIP embeddings "
            "determines contextual relevance. A sliding exclusion queue prevents repetitive "
            "background selection across consecutive synthesis cycles.")

S_M2_NEW = ("Background retrieval in Module M2 treats the scene category as a hard filter and "
            "then matches the requested diurnal phase and weather class against the per-image "
            "metadata record through a deterministic four-tier fallback: a candidate matching "
            "both labels is preferred, failing which one matching the diurnal phase, failing "
            "which one matching the weather class, and failing which any image in the "
            "category. A candidate is drawn uniformly at random from the first tier that is "
            "non-empty, and a sliding exclusion queue holds the sixteen most recently used "
            "backgrounds out of the eligible set, so that no background recurs within "
            "sixteen consecutive retrievals whenever the tier holds at least that many "
            "candidates, while a tier smaller than the window is traversed without "
            "repetition until it is exhausted, after which the queue "
            "recycles its entries; the "
            "retrieval is deterministic in its matching logic but stochastic in its "
            "choice. The fallback tier actually selected and the number of candidates it "
            "contained are reported with each retrieval, so that the selection can be audited "
            "after the fact.")

S_P41_META_OLD = ("Each photograph carries source, attribution, category, diurnal-phase and "
                  "weather metadata, providing")

S_P41_META_NEW = ("Each photograph carries category metadata, and 37,559 of the 37,852 "
                  "(99.2%) additionally carry source, attribution, diurnal-phase and weather "
                  "records (Section 3.3.1), providing")

S_P41_SAMP_OLD = ("and five weather classes (33-34 pairs per category). The batch was "
                  "synthesized")

S_P41_SAMP_NEW = ("and five weather classes (33-34 pairs per category). Within each category "
                  "the host photograph was drawn uniformly at random, without repetition, from "
                  "that category's pool, and the diurnal phase and weather class named above "
                  "are the conditioning labels requested by the sampling plan - assigned in a "
                  "balanced cycle - rather than properties retrieved from the photograph's own "
                  "metadata record; the benchmark therefore exercises the geometric and "
                  "photometric stages of the pipeline rather than the metadata-matching "
                  "retrieval described for Module M2 in Section 3.3.1. The batch was "
                  "synthesized")

S_P46_PROXY_OLD = ("and on that proxy the more strongly harmonised arms score lower "
                   "(Section 4.5.8)")

S_P46_PROXY_NEW = ("and on that proxy the more strongly harmonised build of the same 300 "
                   "conditioning points does not score higher (Section 4.5.8)")

S_P46_TAIL_OLD = ("This negative result is therefore reported at full strength rather than "
                  "softened - on the measurement available to us, the more strongly harmonised "
                  "samples are the less detector-salient ones - and the direction of the "
                  "effect, not its magnitude, is what we interpret.")

S_P46_TAIL_NEW = ("This negative result is therefore reported at full strength rather than "
                  "softened: on the measurement available to us the boundary-harmonisation "
                  "correction of Section 3.4.3 shifts both proxy rates downwards over the "
                  "identical 300 conditioning points, but neither shift reaches significance "
                  "(exact paired McNemar p = 0.14 for the any-class variant and p = 0.05 for "
                  "R_aerial), so the shift is reported descriptively and no directional "
                  "conclusion is drawn from it.")


def _halofix_pair():
    """同一 300 条件点的配对 A/B：未修正 vs 边界协调修正（离线后处理）。

    源：experiments/bcd/out_downstream_halofix/downstream.json
        out_D（修正前渲染） vs out_D_halofix（对同一批渲染施加修正）
    """
    p = Path(__file__).resolve().parent / "out_downstream_halofix" / "downstream.json"
    d = _load(p)
    if not d:
        return None
    A = {r.get("sample_id"): r for r in (dig(d, "out_D.rows") or [])}
    B = {r.get("sample_id"): r for r in (dig(d, "out_D_halofix.rows") or [])}
    common = sorted(set(A) & set(B))
    if not common:
        return None
    out = {"n": len(common)}
    for lbl, key in (("any", "any_hit"), ("aerial", "aerial_hit")):
        a = [1 if A[i].get(key) else 0 for i in common]
        b = [1 if B[i].get(key) else 0 for i in common]
        b10 = sum(1 for i in range(len(common)) if a[i] and not b[i])
        b01 = sum(1 for i in range(len(common)) if b[i] and not a[i])
        pv, nd = mcnemar_exact(b10, b01)
        out[lbl] = {"pre": sum(a) / len(common), "post": sum(b) / len(common),
                    "p": pv, "n_disc": nd, "b10": b10, "b01": b01}
    return out


def _s_41_halo(m):
    """第三次 Implementation revision：边界协调修正 + 配对 proxy 位移。"""
    h = _halofix_pair()
    if not h:
        return None
    a, r = h["any"], h["aerial"]
    return (
        "Implementation revision (boundary harmonisation). The initial build replaced the "
        "host background over the entire extent of the feather band, so residual "
        "high-frequency texture from the repainted crop was carried into the ring immediately "
        "surrounding the target; on the benchmark pairs that ring showed roughly three times "
        "the local high-frequency energy of the surrounding background. The current build "
        "restricts that replacement in the manner described in Section 3.4.3: outside the "
        "silhouette the high-frequency component of the repainted crop is taken from the "
        "pre-repaint composite and the band is re-aligned to it in per-channel mean and "
        "standard deviation. Applied as an offline post-process to the identical 300 "
        "conditioning points, the correction lowers both downstream proxy rates (the any-class "
        f"variant from {f3(a['pre'])} to {f3(a['post'])} and R_aerial from {f3(r['pre'])} to "
        f"{f3(r['post'])}), with exact paired McNemar p = {a['p']:.2f} and {r['p']:.2f} "
        "respectively: the direction is consistent across both variants, but neither shift is "
        "significant, so the correction is presented as a fidelity correction to the host "
        "background rather than as a quality or detection improvement. A paired visual A/B "
        "over 16 conditioning points likewise failed to resolve whether the correction "
        "improves perceived boundary quality, and the softer setting was retained. Every value "
        "reported in Table 6 and Section 4.5.8 is from the build that includes the correction, "
        "and the uncorrected artefacts are retained in the project repository for provenance "
        "only.")


# ═══════════════════════════════════════════════════════════════════════
# B) 数据驱动修正 —— 锚点替换（旧串逐字来自 v8 原文）
# ═══════════════════════════════════════════════════════════════════════
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def pc(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _sig(m, path):
    return m.dp(*path.split(".")) if m.d else None


# ═══════════════════════════════════════════════════════════════════════
# B) 数据驱动修正 —— 锚点（旧串逐字取自 v8 原文）
# ═══════════════════════════════════════════════════════════════════════
def _a_abs_pos(m):
    v = _sig(m, "alignment.pos_err_px.mean")
    s = _sig(m, "alignment.pos_err_px.std")
    d = m.median_diag()
    if v is None or s is None or not d:
        return None
    return f"{f3(v)} +/- {f3(s)} pixels ({100 * v / d:.3f}% of the canvas diagonal)"


def _a_abs_scale(m):
    v = _sig(m, "alignment.scale_err.mean")
    s = _sig(m, "alignment.scale_err.std")
    if v is None or s is None:
        return None
    return f"{f3(v)} +/- {f3(s)}, with cross-modal"


# ── 派生百分比（⑩ 画布对角口径 / half-pixel 越界率）──
HALF_PX_BOUND = 0.5          # 包围盒整数像素量化 → 半像素界


def _worst_rel_pct(m):
    """最差单样本相对位置误差（% 其自身画布对角）。"""
    ss = m.d_samples
    if not ss:
        return None
    try:
        return 100.0 * max(s["alignment"]["pos_err_px"] / math.hypot(s["W"], s["H"])
                           for s in ss)
    except Exception:
        return None


def _g_diag_pair(m):
    """§4.3：均值位移占『中位画布对角』的百分比 + 最差单样本百分比。"""
    v = _sig(m, "alignment.pos_err_px.mean")
    d = m.median_diag()
    w = _worst_rel_pct(m)
    if v is None or not d or w is None:
        return None
    return f"{100 * v / d:.3f}% of the image diagonal (worst case {w:.3f}%)"


def _g_diag_i(m):
    """§5 结论：同一口径的短句形式。"""
    v = _sig(m, "alignment.pos_err_px.mean")
    d = m.median_diag()
    if v is None or not d:
        return None
    return f"i.e. {100 * v / d:.3f}% of the image diagonal"


def _g_beyond_half_px(m):
    """§4.3：残差超过半像素量化界的样本占比。"""
    ss = m.d_samples
    if not ss:
        return None
    pe = [s["alignment"]["pos_err_px"] for s in ss]
    if not pe:
        return None
    k = sum(1 for e in pe if e > HALF_PX_BOUND)
    return f"in {100 * k / len(pe):.1f}% of the samples"


def _a_abs_proxy(m):
    a = m.ds("out_E1_full", "R_aerial")
    b = m.ds("out_E1_paste", "R_aerial")
    if a is None or b is None:
        return None
    return f"({f3(a)} versus {f3(b)})"


def _a_abs_loop(m):
    R = m.bcy_rate("A1")
    if not R or any(r is None for r in R):
        return None
    # A1 四轮实际是持平（0.583 / 0.583 / 0.567 / 0.567），不是「raise」，
    # 故改述为 plateau；配合 abs_raise 锚点（raise → hold）使用。
    return f"at a plateau of {min(R[1:]):.3f}-{max(R[1:]):.3f}"


def _a_p41_usecfg(m):
    return ("20 diffusion steps for target synthesis and 15 repainting steps at a "
            "strength of 0.65, the latter executing 9 denoising steps")


def _a_p41_lat(m):
    L = m.lat()
    if not L:
        return None
    return (f"{f2(L['mean'])} s on average (median {f2(L['median'])} s, range "
            f"{f2(L['min'])}-{f2(L['max'])} s over the {L['n']}-pair batch)")


def _a_p41_lat_tod(m):
    T = m.lat_tod()
    if len(T) < 3:
        return None
    out = []
    for d, prep in (("dawn", "at dawn"), ("morning", "in the morning"),
                    ("afternoon", "in the afternoon"), ("dusk", "at dusk"),
                    ("night", "at night")):
        if d in T:
            out.append(f"{f2(T[d])} s {prep}")
    return ", ".join(out)


def _a_p43_pos(m):
    v = _sig(m, "alignment.pos_err_px.mean")
    if v is None:
        return None
    return (f"{f3(v)} +/- {f3(_sig(m, 'alignment.pos_err_px.std'))} pixels (range "
            f"{f3(_sig(m, 'alignment.pos_err_px.min'))}-"
            f"{f3(_sig(m, 'alignment.pos_err_px.max'))} px)")


def _a_p43_iou(m):
    v = _sig(m, "alignment.iou.mean")
    if v is None:
        return None
    return (f"{f3(v)} +/- {f3(_sig(m, 'alignment.iou.std'))} (range "
            f"{f3(_sig(m, 'alignment.iou.min'))}-{f3(_sig(m, 'alignment.iou.max'))})")


def _a_p43_scale(m):
    v = _sig(m, "alignment.scale_err.mean")
    s = _sig(m, "alignment.scale_err.std")
    x = _sig(m, "alignment.scale_err.max")
    if v is None or s is None or x is None:
        return None
    return f"{f3(v)} +/- {f3(s)}, with an observed maximum of {x * 100:.1f}%"


def _a_p43_rates(m):
    a = _sig(m, "alignment.iou_above_0p5_rate")
    b = _sig(m, "alignment.sub_pixel_rate")
    c = _sig(m, "alignment.scale_below_3pct_rate")
    if a is None or b is None or c is None:
        return None
    return (f"{pc(a)} of samples achieve an IoU of at least 0.5, {pc(b)} reach sub-pixel "
            f"centroid accuracy below one pixel, and {pc(c)} hold the relative scale error "
            f"within 3%")


def _a_p43_split(m):
    small, big = m.split_by_side()
    if not small or not big:
        return None
    return (f"the {len(small)} samples whose target bounding box has a side shorter than 40 "
            f"pixels reach a mean IoU of {f3(m.agg(small, 'iou'))} against "
            f"{f3(m.agg(big, 'iou'))} for the remaining {len(big)}, and their mean relative "
            f"scale error is {100 * m.agg(small, 'scale_err'):.1f}% against "
            f"{100 * m.agg(big, 'scale_err'):.1f}%")


def _a_p43_resid(m):
    small, big = m.split_by_side()
    if not small or not big:
        return None
    return (f"it averages {f2(m.agg(small, 'pos_err_px'))} px for targets below 40 pixels "
            f"against {f2(m.agg(big, 'pos_err_px'))} px above that size")


def _a_p43_tod(m):
    T = m.tod_stat("pos_err_px")
    I = m.tod_stat("iou")
    if len(T) < 5 or len(I) < 5:
        return None
    order = ("dawn", "morning", "afternoon", "dusk", "night")
    prep = {"dawn": "at dawn", "morning": "in the morning", "afternoon": "in the afternoon",
            "dusk": "at dusk", "night": "at night"}
    body = ", ".join(f"{T[d]:.3f} px {prep[d]}" for d in order)
    return (f"the mean position error varied by only {max(T.values()) - min(T.values()):.3f} px "
            f"({body}) and the mean IoU stayed within {min(I.values()):.3f}-{max(I.values()):.3f}")


def _a_p43_bris_abs(m):
    mu = _sig(m, "quality.brisque.mean")
    sd = _sig(m, "quality.brisque.std")
    pr = _sig(m, "quality.pass_rate")
    n = _sig(m, "n")
    if mu is None or sd is None or pr is None or not n:
        return None
    return (f"mean BRISQUE score of {f2(mu)} (median {f2(m.d_median('brisque'))}, standard "
            f"deviation {f2(sd)}), with {int(round(pr * n))} of the {n} samples ({pc(pr)}) passing")


def _a_p43_bris_scene(m):
    S = m.scene_brisque()
    T = m.tod_brisque()
    if not S or not T:
        return None
    best_s = min(S, key=S.get)
    best_t = min(T, key=T.get)
    worst_s = max(S, key=S.get)
    worst_t = max(T, key=T.get)
    return (f"{best_s.replace('_', '-')} backgrounds (mean BRISQUE {f2(S[best_s])}) and "
            f"{best_t} scenes ({f2(T[best_t])}), whereas the worst were "
            f"{worst_s.replace('_', '-')} backgrounds ({f2(S[worst_s])}) and {worst_t} scenes "
            f"({f2(T[worst_t])})")


def _a_p43_entropy(m):
    nu = _sig(m, "diversity.entropy_nu_bits")
    nv = _sig(m, "diversity.entropy_nv_bits")
    sc = _sig(m, "diversity.entropy_scale_bits")
    if nu is None or nv is None or sc is None:
        return None
    return (f"entropies of {f3(nu)} bits for horizontal position, {f3(nv)} bits for vertical "
            f"elevation and {f3(sc)} bits for target scale")


def _a_e1_geom(m):
    a, b = m.e1("full"), m.e1("paste")
    if not a or not b:
        return None
    return (f"the mean centroid position error is {f3(a.get('pos_err_mean_px'))} px with "
            f"repainting against {f3(b.get('pos_err_mean_px'))} px with hard paste, the mean "
            f"bounding-box IoU is {f3(a.get('iou_mean'))} and {f3(b.get('iou_mean'))}, and the "
            f"mean BRISQUE score differs by "
            f"{abs(a.get('brisque_mean', 0) - b.get('brisque_mean', 0)):.2f} "
            f"({f2(a.get('brisque_mean'))} with repainting versus {f2(b.get('brisque_mean'))} "
            f"with hard paste), with quality pass rates of {pc(a.get('brisque_pass_rate'))} and "
            f"{pc(b.get('brisque_pass_rate'))} respectively")


def _a_e1_lcr(m):
    if m.fus_p("lcr") is None or m.fus_p("rlj") is None:
        return None
    return (f"Within the {m.fus_p('lcr', 'n')}-pair matched set, repainting moves the LCR from "
            f"{f3(m.fus('out_E1_paste', 'lcr'))} to {f3(m.fus('out_E1_full', 'lcr'))} and the "
            f"RLJ from {f3(m.fus('out_E1_paste', 'rlj'))} to "
            f"{f3(m.fus('out_E1_full', 'rlj'))}, with paired Wilcoxon signed-rank p = "
            f"{m.fus_p('lcr'):.3g} and p = {m.fus_p('rlj'):.3g} respectively")


def _a_e1_bge(m):
    if m.fus_p("bge") is None:
        return None
    return (f"the boundary-gradient excess moves from {f3(m.fus('out_E1_paste', 'bge'))} under "
            f"hard paste to {f3(m.fus('out_E1_full', 'bge'))} under repainting (paired "
            f"Wilcoxon p = {m.fus_p('bge'):.3g})")


def _a_e1_proxy(m):
    a = m.ds("out_E1_full", "R_aerial")
    b = m.ds("out_E1_paste", "R_aerial")
    if a is None or b is None:
        return None
    return (f"R_aerial is {f3(a)} for repainting versus {f3(b)} for hard paste, a factor of "
            f"{b / max(a, 1e-9):.1f}, and the median best-box IoU is "
            f"{f3(m.ds('out_E1_full', 'iou_best_median'))} versus "
            f"{f3(m.ds('out_E1_paste', 'iou_best_median'))}")


def _a_e3_geom(m):
    a, b = m.e3("lora"), m.e3("nolora")
    if not a or not b:
        return None
    return (f"mean IoU {f3(a.get('iou_mean'))} versus {f3(b.get('iou_mean'))}; position error "
            f"{f3(a.get('pos_err_mean_px'))} versus {f3(b.get('pos_err_mean_px'))} px) and "
            f"perceptual quality is equivalent (BRISQUE {f2(a.get('brisque_mean'))} versus "
            f"{f2(b.get('brisque_mean'))}, quality pass rate "
            f"{pc(a.get('brisque_pass_rate'))} in both arms)")


def _a_e3_proxy(m):
    a = m.ds("out_E3_lora", "R_aerial")
    b = m.ds("out_E3_nolora", "R_aerial")
    if a is None or b is None:
        return None
    word = "higher" if a >= b else "lower"
    return (f"the adapter-mounted arm scores {word} (R_aerial {f3(a)} versus {f3(b)} for the "
            f"no-adapter arm)")


def _e4trio(m):
    return m.e4("both"), m.e4("depth"), m.e4("seg")


def _e4fus(m, metric):
    return (m.fus("out_E4_both", metric, which="E4"),
            m.fus("out_E4_depth", metric, which="E4"),
            m.fus("out_E4_seg", metric, which="E4"))


def _a_e4_geom(m):
    a, b, c = _e4trio(m)
    if not (a and b and c):
        return None
    return (f"(mean IoU {f3(a.get('iou_mean'))}, {f3(b.get('iou_mean'))} and "
            f"{f3(c.get('iou_mean'))}, position error {f3(a.get('pos_err_mean_px'))}, "
            f"{f3(b.get('pos_err_mean_px'))} and {f3(c.get('pos_err_mean_px'))} px)")


def _a_e4_quality(m):
    a, b, c = _e4trio(m)
    if not (a and b and c):
        return None
    return (f"(BRISQUE {f2(a.get('brisque_mean'))}, {f2(b.get('brisque_mean'))} and "
            f"{f2(c.get('brisque_mean'))})")


def _a_e4_bge(m):
    v = _e4fus(m, "bge")
    p1 = m.fus_p("bge", which="E4", pair=("both", "depth"))
    p2 = m.fus_p("bge", which="E4", pair=("both", "seg"))
    if v[0] is None or p1 is None or p2 is None:
        return None
    return (f"BGE {f3(v[0])} against {f3(v[1])} for depth-only and {f3(v[2])} for "
            f"segmentation-only; paired Wilcoxon over {m.e4('n') or 100} matched pairs, p = "
            f"{p1:.3g} and p = {p2:.3g}")


def _a_e4_lcr(m):
    v = _e4fus(m, "lcr")
    if v[0] is None:
        return None
    return f"({f3(v[0])} against {f3(v[1])} and {f3(v[2])})"


def _a_e4_rlj(m):
    v = _e4fus(m, "rlj")
    p1 = m.fus_p("rlj", which="E4", pair=("both", "depth"))
    p2 = m.fus_p("rlj", which="E4", pair=("both", "seg"))
    if v[0] is None or p1 is None or p2 is None:
        return None
    return f"({f3(v[0])} against {f3(v[1])} and {f3(v[2])}; p = {p1:.3g} and p = {p2:.3g})"


def _a_e4_proxy(m):
    rs = [m.ds(d, "R_aerial") for d in
          ("out_E4_both", "out_E4_depth", "out_E4_seg")]
    if any(r is None for r in rs):
        return None
    return (f"(R_aerial {f3(max(rs))} for the best-scoring arm versus {f3(min(rs))} for the "
            f"worst-scoring arm)")


# ═══════════════════════════════════════════════════════════════════════
# B2) 论断句修正（2026-10-08）
#
# 背景：v8 时代 repainting 的 R_aerial = 0.153（远低于 hard paste 的 0.540），
# 据此把「边界融合 ↑ ⇒ 检测显著性 ↓」写成一条 trade-off。halo 融合带修复（v10）
# 后该数字反转为 0.540 vs 0.507，配对 McNemar 不显著（p ≈ 0.49）；E3 / E4 的
# proxy 对比同样不显著（p = 0.26 / 0.12 / 0.82）。因此下列论断全部失效：
#   · "exposes a cost" / "markedly less salient" / "a trade-off that we report
#     rather than resolve" / "systematically undervalues" / "in tension" /
#     "favours hard paste over repainting" / "less salient target ... (0.153 versus 0.540)"
# 统一改为 null result：COCO 二值检出代理太粗，分辨不出像素级已经分辨出的差异。
# 保留的论点：COCO-class proxy 不适合评估边界融合（理由改为「连像素级差异都
# 分辨不出，且受背景假阳性污染」）。
# 附带修正：BGE 不再是 cost（2.153 vs 2.253，p = 0.96）；A1 闭环不是「raise」而是持平。
# ═══════════════════════════════════════════════════════════════════════
def _a_abs_cost(m):
    """Abstract [3.6]：BGE 从「小代价」改为「无显著变化」。"""
    return ("with no significant change in boundary-gradient excess and no detectable "
            "difference under a COCO-class detector proxy ")


def _a_abs_tradeoff(m):
    """Abstract [3.6]：trade-off → null result。"""
    return (" - a null result that we report as evidence that such a proxy is too coarse to "
            "value boundary fusion - ")


def _a_abs_raise(m):
    """Abstract [3.6]：A1 四轮不是「raise」0.583，而是持平在 0.567-0.583。"""
    return "hold per-attempt acceptance"


def _a_e4_bge_frame(m):
    """[181.4] 首句：实测 seg-only 的 BGE 最高（2.316），dual 居中（2.286）。"""
    return ("The dual-condition arm does not carry the lowest boundary-gradient excess, and "
            "the ordering it does show is not the one the rationale implies")


def _a_e4_rlj_frame(m):
    """[181.5]：both 的 RLJ 反而最高（0.252 vs 0.236 / 0.249），与 depth 差异显著。

    锚点只覆盖不加数字的前半句，数值仍由 e4_rlj 锚点提供，两者互不冲突。
    """
    return "The ring-band luminance jump does not favour the dual-condition arm either"


def _a_e4_proxy_frame(m):
    """[181.7]：实测 dual-condition 最高（0.540），depth-only 最低（0.460），三臂不可分。"""
    return ("The detector proxy does not separate the three arms either, and the ordering it "
            "does show follows the dual-condition arm rather than the single-condition ones")


def _a_b_caveat_frame(m):
    """[190.4]：proxy 分辨不出 hard paste 与 repainting，不能再称其偏袒 hard paste。"""
    return ("the same coarse quantity that fails to separate the hard-paste and repainting "
            "arms in Section 4.5.1")


def _a_proxy_ctrl_frame(m):
    """[192.3]：32% 的背景 any-class 假阳性本身已不可忽略（原句称其高于所有合成臂，错）。"""
    return ("a level that is not negligible and that we therefore discard as "
            "non-discriminative")


def _r_e1_proxy_full(m):
    """整段重写 v8 [172]（v9 [174]）：trade-off → null result。"""
    a = m.ds("out_E1_full", "R_aerial")
    b = m.ds("out_E1_paste", "R_aerial")
    if a is None or b is None:
        return None
    st = m.ds_pair("out_E1_full", "out_E1_paste") or {}
    p, d, n = st.get("p"), st.get("delta"), st.get("n")
    ia = m.ds("out_E1_full", "iou_best_median")
    ib = m.ds("out_E1_paste", "iou_best_median")
    stat = ""
    if p is not None and d is not None:
        stat = (f", and a paired McNemar test over the {n} matched points does not separate "
                f"them (difference {d:+.3f}, p = {p:.2f})")
    return (
        "The same comparison also bears on the detector proxy, and here the result is a null "
        "rather than a cost. Under the detector proxy of Section 4.2 the two arms are "
        f"indistinguishable: R_aerial is {f3(a)} for repainting against {f3(b)} for hard "
        f"paste, with a median best-box IoU of {f3(ia)} versus {f3(ib)}{stat}. Because the "
        "geometric alignment of the two arms is identical, any difference would be "
        "attributable to the repainting step itself, and its absence means that the pipeline "
        "carries no detectable saliency penalty relative to a hard composite. Read as a "
        "controlled ablation, the pair therefore quantifies a tie under this instrument "
        "rather than a ranking of the two synthesis strategies. The implication is narrow but "
        "useful: a COCO-class proxy is too coarse to value the mechanism at all, in either "
        "direction, while it does respond strongly to background content (Section 4.5.8). We "
        "do not claim that the full pipeline improves downstream detectability; the evidence "
        "is inconclusive in either direction, and a drone-specific detector evaluation, as "
        "identified in Section 4.6, is required to settle the question."
    )


def _r_proxy_block(m):
    """整段重写 v8 [190]（v9 [192]）：proxy 排序叙事 → 不可区分 + 工具太粗。"""
    bg = m.ds("out_D", "fpr.fp_aerial_rate")
    bgany = m.ds("out_D", "fpr.any_det_rate")
    if bg is None or bgany is None:
        return None
    keys = ("out_E1_full", "out_E1_paste", "out_E3_lora", "out_E3_nolora",
            "out_E4_both", "out_E4_depth", "out_E4_seg")
    ra = {d: m.ds(d, "R_aerial") for d in keys}
    if any(v is None for v in ra.values()):
        return None
    ps = [m.ds_p("out_E1_full", "out_E1_paste"),
          m.ds_p("out_E3_lora", "out_E3_nolora"),
          m.ds_p("out_E4_both", "out_E4_depth"),
          m.ds_p("out_E4_both", "out_E4_seg")]
    ps = [x for x in ps if x is not None]
    ptxt = (f"none of the corresponding paired McNemar tests on R_aerial reaches significance "
            f"(smallest p = {min(ps):.2f})" if ps else
            "no paired test on R_aerial reaches significance")
    return (
        "The practical purpose of the dataset is downstream detection, so usability was "
        "monitored with the detector proxy defined in Section 4.2. Table 6 reports R_aerial "
        "and the best-box IoU for the primary benchmark, the E1 arms, the domain-adaptation "
        "arms, the conditional-dimensionality arms and, as a control, the un-manipulated "
        "background images of the benchmark. The control fixes the interpretation: "
        f"backgrounds alone already trigger R_aerial = {f3(bg)}, and the weaker any-class "
        f"variant fires on {pc(bgany)} of empty backgrounds, "
        + _a_proxy_ctrl_frame(m) + ". "
        "The proxy therefore measures the co-occurrence of a generic aerial class rather "
        "than drone detection, and the R_aerial contrasts it produces are correspondingly "
        "uninformative: the arms are not separable, with R_aerial at "
        f"{f3(ra['out_E1_full'])} against {f3(ra['out_E1_paste'])} for repainting versus "
        f"hard paste, {f3(ra['out_E3_lora'])} against {f3(ra['out_E3_nolora'])} for the "
        f"adapter arms, and {f3(ra['out_E4_both'])}, {f3(ra['out_E4_depth'])} and "
        f"{f3(ra['out_E4_seg'])} across the conditional-dimensionality arms, and {ptxt}. On "
        "the same samples a continuous localisation measure is more sensitive - the mean "
        "best-box IoU separates the dual-condition and depth-only arms of Section 4.5.4 "
        "where the binary rate does not - which indicates that the coarseness of the "
        "instrument, not the absence of a mechanism effect, is what limits this benchmark. "
        "We therefore report the null result as a finding about the evaluation instrument "
        "rather than about the pipeline, and we refrain from any claim that the framework "
        "improves downstream detectability."
        + _halo_proxy_note(m)
    )


def _halo_proxy_note(m):
    """§4.5.8 追加：边界协调修正的配对下游位移（同一 300 条件点）。"""
    h = _halofix_pair()
    if not h:
        return ""
    a, r = h["any"], h["aerial"]
    return (
        " Because the boundary-harmonisation correction of Section 3.4.3 was introduced after "
        f"the first downstream evaluation, the same {h['n']} conditioning points were re-scored "
        f"before and after it: the any-class variant falls from {f3(a['pre'])} to "
        f"{f3(a['post'])} and R_aerial from {f3(r['pre'])} to {f3(r['post'])}, with exact "
        f"paired McNemar p = {a['p']:.2f} and {r['p']:.2f} respectively. The direction is "
        "consistent across both proxy variants, but neither shift is significant and the "
        "binary instrument does not resolve it, so the change is reported as a bookkeeping "
        "fact about the build rather than as an effect; every value in Table 6 is from the "
        "build that includes the correction."
    )


def _r_sec49_block(m):
    """整段重写 v8 [192]（v9 [194]）：trade-off / in tension → null result。"""
    st = m.ds_pair("out_E1_full", "out_E1_paste")
    if not st:
        return None
    d, p = st.get("delta"), st.get("p")
    stat = (f"difference {d:+.3f}, paired McNemar p = {p:.2f}"
            if p is not None and d is not None else "the paired test does not reach significance")
    return (
        "Within the E1 pairing the same null appears in a second, independent form: the arm "
        "that is less fused by every pixel-level measure is neither easier nor harder for the "
        "proxy to detect, and with identical geometry the paired difference does not reach "
        f"significance ({stat}). We accordingly state only the claim the data support - that "
        "a COCO-class proxy is too coarse an instrument to value boundary fusion, since it "
        "fails to separate arms that every pixel-level boundary measure distinguishes "
        "decisively - and we explicitly withhold both claims the data cannot support, namely "
        "that the full pipeline improves downstream detectability, and that boundary fusion "
        "penalises it."
    )


def _a_concl_fourth(m):
    """§5 [212] 第四条 finding：更新 LCR/RLJ/BGE 与循环数字，trade-off → null。"""
    lcr_b = m.fus("out_E1_paste", "lcr")
    lcr_r = m.fus("out_E1_full", "lcr")
    rlj_b = m.fus("out_E1_paste", "rlj")
    rlj_r = m.fus("out_E1_full", "rlj")
    bge_b = m.fus("out_E1_paste", "bge")
    bge_r = m.fus("out_E1_full", "bge")
    if None in (lcr_b, lcr_r, rlj_b, rlj_r, bge_b, bge_r):
        return None
    st = m.ds_pair("out_E1_full", "out_E1_paste") or {}
    a = m.ds("out_E1_full", "R_aerial")
    b = m.ds("out_E1_paste", "R_aerial")
    p = st.get("p")
    R = m.bcy_rate("A1")
    looptxt = (f"four closed-loop cycles hold per-attempt acceptance at "
               f"{min(R[1:]):.3f}-{max(R[1:]):.3f} without further monotonic improvement"
               if R else "four closed-loop cycles show no further monotonic improvement")
    proxytxt = (f"and the two strategies are statistically indistinguishable under a "
                f"COCO-class detector proxy (R_aerial {f3(a)} versus {f3(b)}, paired "
                f"p = {p:.2f}), a null result " if p is not None and a is not None
                else "and a COCO-class detector proxy does not separate the two strategies, ")
    return (
        "Fourth, paired ablations quantify an honest effect at the pixel level but a null "
        "under the detector proxy: depth-guided repainting improves the luminance continuity "
        f"of the boundary relative to hard paste (LCR {f3(lcr_b)} versus {f3(lcr_r)}; RLJ "
        f"{f3(rlj_b)} versus {f3(rlj_r)}; p < 1e-4), with no significant change in gradient "
        f"excess (BGE {f3(bge_b)} versus {f3(bge_r)}; p = {m.fus_p('bge'):.2f}), "
        + proxytxt +
        "that we report as evidence that such a proxy is too coarse to value boundary "
        f"fusion, and {looptxt}, while the second ablation finds that supplying both geometric "
        "conditions together did not improve boundary harmonisation."
    )


def _a_e5_cos(m):
    a, b = m.e5("fixed"), m.e5("indep")
    if not a or not b:
        return None
    ra = a.get("cosine_mean_across_groups")
    rb = b.get("cosine_mean_across_groups")
    return (f"The mean cosine similarity between the target signatures of consecutive frames is "
            f"{f3(ra)} +/- {f3(a.get('cosine_std'))} with a fixed seed against {f3(rb)} +/- "
            f"{f3(b.get('cosine_std'))} with independent seeds, a difference of roughly "
            f"{ra / max(rb, 1e-9):.0f}fold.")


def _e6_curve_vals(m):
    return tuple(dig(m.e6(str(n)), "qc_mean_best") for n in (1, 2, 3))


def _a_e6_curve(m):
    v = _e6_curve_vals(m)
    if any(x is None for x in v):
        return None
    return (f"Mean best-of-N quality rises from {v[0]:.4f} at N = 1 to {v[1]:.4f} at N = 2 and "
            f"{v[2]:.4f} at N = 3, at a cost of exactly N generation attempts per target.")


def _a_e6_marginal(m):
    v = _e6_curve_vals(m)
    if any(x is None for x in v):
        return None
    return (f"the second candidate buys {100 * (v[1] - v[0]):.1f} points of quality for one "
            f"extra generation, whereas the third buys {100 * (v[2] - v[1]):.1f}")


def _s9_share(m):
    r = m.bcy("A1")
    if not r:
        return None
    n9 = nf = 0
    for cy in r:
        for rd in (cy.get("rounds") or []):
            for k, v in (rd.get("failure_hist") or {}).items():
                if k == "PASS":
                    continue
                nf += v
                if "UNDETECTABLE" in k.upper():
                    n9 += v
    return (n9 / nf) if nf else None


def _a_b_a1(m):
    R = m.bcy_rate("A1")
    tot = m.bcy_tot("A1")
    if not R or any(r is None for r in R) or tot[0] is None:
        return None
    return ("In A1 the per-attempt acceptance rate runs "
            + ", ".join(f"{r:.3f}" for r in R)
            + f" over the four cycles, i.e. it reaches a plateau between {min(R[1:]):.3f} and "
              f"{max(R[1:]):.3f} and thereafter remains close to flat; the cumulative totals are "
              f"{tot[0]} attempts and {tot[1]} accepted samples.")


def _a_b_a2(m):
    R = m.bcy_rate("A2")
    tot = m.bcy_tot("A2")
    rA1 = m.bcy_rate("A1")
    tA1 = m.bcy_tot("A1")
    if not R or any(r is None for r in R) or not rA1 or not tA1[0]:
        return None
    return (f"delivering {min(R):.2f}-{max(R):.2f} of the requested quota per cycle instead of "
            f"{min(rA1):.2f}, at the cost of {tot[0] / max(tA1[0], 1):.1f} times as many "
            f"generation attempts ({tot[0]} attempts for {tot[1]} accepted samples in A2)")


def _a_b_a3(m):
    cur = m.bcy("A3")
    R = m.bcy_rate("A3")
    tot = m.bcy_tot("A3")
    if not cur or not R or any(r is None for r in R):
        return None
    A = [x.get("attempts") for x in cur]
    return (f"reduces the attempts required per cycle from {max(A)} to between {min(A[1:])} and "
            f"{max(A[1:])} while holding target achievement near {sum(R) / len(R):.2f}, which is "
            f"the clearest efficiency effect observed in the loop; the cumulative totals for A3 "
            f"are {tot[0]} attempts and {tot[1]} accepted samples.")


def _a_b_caveat(m):
    sh = _s9_share(m)
    if sh is None:
        return None
    return f"in the A1 failure log {100 * sh:.0f}% of rejections carry the undetectable-target code"


ANCHORS_D = [
    ("abs_pos", "1.212 +/- 0.346 pixels (0.071% of the canvas diagonal)", _a_abs_pos),
    ("abs_scale", "0.0268 +/- 0.0152, with cross-modal", _a_abs_scale),
    ("abs_proxy", "(0.153 versus 0.540)", _a_abs_proxy),
    ("abs_loop", "from 0.150 to a plateau of 0.217-0.233", _a_abs_loop),
    ("p41_usecfg",
     "20 diffusion steps for target synthesis and 15 blending steps at a blending strength of 0.65",
     _a_p41_usecfg),
    ("p41_lat", "17.2 s on average (median 16.7 s, range 14.1-24.9 s over the 300-pair batch)",
     _a_p41_lat),
    ("p41_lat_tod",
     "16.96 s at dawn, 17.30 s in the morning, 17.26 s in the afternoon, 17.13 s at dusk and 17.45 s at night",
     _a_p41_lat_tod),
    ("p43_pos", "1.212 +/- 0.346 pixels (range 0.190-2.206 px)", _a_p43_pos),
    ("p43_iou", "0.7028 +/- 0.1330 (range 0.321-0.961)", _a_p43_iou),
    ("p43_scale",
     "0.0268 +/- 0.0152, with an observed maximum of 10.0%", _a_p43_scale),
    ("p43_rates",
     "91.7% of samples achieve an IoU of at least 0.5, 28.7% reach sub-pixel centroid accuracy below one pixel, and 62.7% hold the relative scale error within 3%",
     _a_p43_rates),
    ("p43_split",
     "the 149 samples whose target occupies fewer than 40 pixels reach a mean IoU of 0.682 against 0.724 for the remaining 151, and their mean relative scale error is 3.9% against 1.4%",
     _a_p43_split),
    ("p43_resid", "it averages 1.23 px for targets below 40 pixels against 1.19 px above that size",
     _a_p43_resid),
    ("p43_tod",
     "the mean position error varied by only 0.044 px (1.216 px at dawn, 1.209 px in the morning, 1.220 px in the afternoon, 1.229 px at dusk and 1.185 px at night) and the mean IoU stayed within 0.696-0.706",
     _a_p43_tod),
    ("p43_bris_abs",
     "mean BRISQUE score of 20.42 (median 14.60, standard deviation 20.72), with 265 of the 300 samples (88.3%) passing",
     _a_p43_bris_abs),
    ("p43_bris_scene",
     "afternoon scenes (mean BRISQUE 16.12) and forest backgrounds (14.17), whereas the worst were pure-sky backgrounds (34.34) and morning scenes (25.23)",
     _a_p43_bris_scene),
    ("p43_entropy",
     "entropies of 3.305 bits for horizontal position, 3.296 bits for vertical elevation and 2.497 bits for target scale",
     _a_p43_entropy),
    ("e1_geom",
     "the mean centroid position error is 1.237 px in both, the mean bounding-box IoU is 0.706 in both, and the mean BRISQUE score differs by 0.03 (20.60 with repainting versus 20.63 with hard paste), with quality pass rates of 88.0% and 88.7% respectively",
     _a_e1_geom),
    ("e1_lcr",
     "Repainting reduces the LCR from 1.656 to 0.811 and the RLJ from 0.247 to 0.170; both reductions are significant under a paired Wilcoxon signed-rank test (p = 3.0 x 10^-8 and 6.3 x 10^-5 respectively)",
     _a_e1_lcr),
    ("e1_bge",
     "the boundary-gradient excess rises from 2.444 under hard paste to 2.546 under repainting (paired Wilcoxon p = 0.019)",
     _a_e1_bge),
    ("e1_proxy",
     "R_aerial is 0.153 for repainting versus 0.540 for hard paste, a factor of 3.5, and the median best-box IoU is 0.000 versus 0.836",
     _a_e1_proxy),
    ("e3_geom",
     "mean IoU 0.707 versus 0.744; position error 1.247 versus 1.249 px) and perceptual quality is equivalent (BRISQUE 20.39 versus 20.35, quality pass rate 89.0% in both arms)",
     _a_e3_geom),
    ("e3_proxy",
     "the adapter-mounted arm scores marginally higher (R_aerial 0.140 versus 0.130 for the no-adapter arm)",
     _a_e3_proxy),
    ("e4_geom", "(mean IoU 0.707 and position error 1.247 px in all three)", _a_e4_geom),
    ("e4_quality", "(BRISQUE 20.39, 20.41 and 20.46)", _a_e4_quality),
    ("e4_bge",
     "BGE 2.631 against 2.272 for depth-only and 2.069 for segmentation-only; paired Wilcoxon over 100 matched pairs, p = 9.8 x 10^-6 and p = 2.8 x 10^-10",
     _a_e4_bge),
    ("e4_lcr", "(0.782 against 0.873 and 0.875)", _a_e4_lcr),
    ("e4_rlj", "(0.178 against 0.183 and 0.185; p = 0.63 and 0.41)", _a_e4_rlj),
    ("e4_proxy", "(R_aerial 0.200 versus 0.110 for depth-only)", _a_e4_proxy),
    ("e5_cos",
     "The mean cosine similarity between the target signatures of consecutive frames is 0.322 +/- 0.116 with a fixed seed against 0.051 +/- 0.030 with independent seeds, a difference of roughly sixfold.",
     _a_e5_cos),
    ("e6_curve",
     "Mean best-of-N quality rises from 0.9234 at N = 1 to 0.9574 at N = 2 and 0.9707 at N = 3, at a cost of exactly N generation attempts per target.",
     _a_e6_curve),
    ("e6_marginal",
     "the second candidate buys 3.4 points of quality for one extra generation, whereas the third buys 1.3",
     _a_e6_marginal),
    ("b_a1_curve",
     "In A1 the per-attempt acceptance rate rises from 0.150 in cycle 0 to 0.233, 0.217 and 0.217, i.e. it reaches a plateau between 0.217 and 0.233 and thereafter remains flat; the cumulative totals are 240 attempts and 49 accepted samples.",
     _a_b_a1),
    ("b_a2_gain",
     "delivering 0.35-0.43 of the requested quota per cycle instead of 0.15, at the cost of three times as many generation attempts (720 attempts for 98 accepted samples in A2)",
     _a_b_a2),
    ("b_a3_gain",
     "reduces the attempts required per cycle from 180 to between 122 and 136 while holding target achievement near 0.40, which is the clearest efficiency effect observed in the loop; the cumulative totals for A3 are 568 attempts and 95 accepted samples.",
     _a_b_a3),
    ("b_caveat", "in the A1 failure log 70% of rejections carry the undetectable-target code",
     _a_b_caveat),
    # ── B2) 论断句修正（2026-10-08，halo 修复后 trade-off 失效）──
    ("abs_cost",
     "at a small cost in boundary-gradient excess but reduces saliency to a COCO-class detector proxy ",
     _a_abs_cost),
    ("abs_tradeoff", " - a trade-off that we report rather than resolve - ", _a_abs_tradeoff),
    ("abs_raise", "raise per-attempt acceptance", _a_abs_raise),
    ("e4_bge_frame",
     "The dual-condition arm carries the highest boundary-gradient excess, not the lowest",
     _a_e4_bge_frame),
    ("e4_rlj_frame",
     "Only the ring-band luminance jump favours the dual-condition arm, and not significantly",
     _a_e4_rlj_frame),
    ("e4_proxy_frame",
     "The detector proxy moves in the opposite direction again, with the segmentation-only arm scoring highest",
     _a_e4_proxy_frame),
    ("b_caveat_frame",
     "the same quantity that favours hard paste over repainting in Section 4.5.1",
     _a_b_caveat_frame),
    ("concl_fourth",
     "Fourth, paired ablations quantify an honest trade-off: depth-guided repainting improves the luminance continuity of the boundary relative to hard paste (LCR 0.811 versus 1.656; RLJ 0.170 versus 0.247; p < 1e-4) at a small cost in gradient excess (BGE 2.546 versus 2.444; p = 0.019) but presents a less salient target to a COCO-class detector proxy (0.153 versus 0.540), and four closed-loop cycles raise per-attempt acceptance from 0.150 to a plateau of 0.217-0.233 without further monotonic improvement, while the second ablation finds that supplying both geometric conditions together did not improve boundary harmonisation.",
     _a_concl_fourth),
]

# ── 整段重写（数据驱动；与锚点独立，在锚点之后执行，故可安全覆盖同段锚点）──
REWRITE_SECTIONS = [
    (P_E1_PROXY, _r_e1_proxy_full),   # v8 [172] → v9 [174]：trade-off → null result
    (P_PROXY, _r_proxy_block),        # v8 [190] → v9 [192]：proxy 排序 → 不可区分
    (P_SEC49, _r_sec49_block),        # v8 [192] → v9 [194]：trade-off → null result
]

_G = {a[0]: a for a in ANCHORS_D}

ANCHOR_MAP = [
    (P_ABS, ["abs_pos", "abs_scale", "abs_proxy", "abs_loop",
             "abs_cost", "abs_tradeoff", "abs_raise"]),
    (P_SEC41_A, ["p41_usecfg", "p41_lat", "p41_lat_tod"]),
    (P_SEC43_A, ["p43_pos", "p43_iou"]),
    (P_SEC43_B, ["p43_scale", "p43_rates", "p43_split", "p43_resid"]),
    (P_SEC43_TOD, ["p43_tod"]),
    (P_SEC43_C, ["p43_bris_abs", "p43_bris_scene", "p43_entropy"]),
    (P_E1_STAT, ["e1_geom"]),
    (P_E1_LCR, ["e1_lcr", "e1_bge"]),
    (P_E1_PROXY, ["e1_proxy"]),
    (P_E3, ["e3_geom", "e3_proxy"]),
    (P_E4, ["e4_geom", "e4_quality", "e4_bge", "e4_lcr", "e4_rlj", "e4_proxy",
            "e4_bge_frame", "e4_rlj_frame", "e4_proxy_frame"]),
    (P_E5, ["e5_cos"]),
    (P_E6, ["e6_curve", "e6_marginal"]),
    (P_B_INTRO, ["b_a1_curve", "b_a2_gain", "b_a3_gain"]),
    (P_B_CAVEAT, ["b_caveat", "b_caveat_frame"]),
    (P_CONCL_B, ["concl_fourth"]),
]

# 全文静态词组替换（先做；随后被整段替换的段落不受影响）
GLOBAL_REPL = [
    ("dual-conditioned ControlNet spatial inpainting",
     "dual-conditioned ControlNet spatial repainting"),
    ("Stable Diffusion 1.5 Inpainting", "Stable Diffusion 1.5 image-to-image"),
    ("latent state blending is enforced throughout reverse diffusion",
     "the repainted crop is blended back in image space"),
    ("gradient-domain Poisson blending", "feathered alpha blending"),
    ("To reduce the computational overhead of full-canvas diffusion",
     "To reduce the computational overhead of localized diffusion"),
    ("its Pearson correlation with target pixel count is -0.06",
     "its Pearson correlation with target pixel count is -0.043"),
    ("1.212 +/- 0.346 px", "1.226 +/- 0.448 px"),
    ("1.212 +/- 0.346 pixels and a relative scale error of 0.0268 +/- 0.0152",
     "1.226 +/- 0.448 pixels and a relative scale error of 0.028 +/- 0.021"),
    ("0.071% of the image diagonal (worst case 0.143%)", _g_diag_pair),
    ("i.e. 0.071% of the image diagonal", _g_diag_i),
    # ⑩ 原 71.3% 无出处且不可复现；实测「残差 > 0.5 px」= 294/300 = 98.0%
    ("in 71.3% of the samples", _g_beyond_half_px),
    # ── 诚信修正（2026-10-11）：正文 ↔ 代码/数据对齐 ──
    (S_M2_OLD, S_M2_NEW),                 # ② M2 检索机制（去 CLIP 过度声明）
    (S_P41_META_OLD, S_P41_META_NEW),     # ③ 背景池元数据覆盖率 37,559/37,852
    (S_P41_SAMP_OLD, S_P41_SAMP_NEW),     # ② benchmark 背景为类内均匀随机
    (S_P46_PROXY_OLD, S_P46_PROXY_NEW),   # ⑧ §4.6 去掉未检验的反向代价表述
    (S_P46_TAIL_OLD, S_P46_TAIL_NEW),     # ⑧ 补 p 值，改为描述性陈述
]

# 有意保留（不依赖本次重跑，需人工确认）
KEEP_MANUAL = [
    "lower the false-acceptance rate by 79.0% relative to a single perceptual threshold",
    "field-level accuracy of 0.985 and an exact schema match rate of 0.93",
    "peak footprint of about 10 GB", "lowering the peak to 4.55 GB",
]

# ═══════════════════════════════════════════════════════════════════════
# docx 写入工具
# ═══════════════════════════════════════════════════════════════════════
def _has_math(p):
    return any(el.tag.endswith("}oMath") or el.tag.endswith("}oMathPara")
               for el in p._p.iter())


def _strip_math(p):
    for el in list(p._p.iter()):
        if el.tag.endswith("}oMath") or el.tag.endswith("}oMathPara"):
            par = el.getparent()
            if par is not None:
                par.remove(el)


def _set_run_text(p, text):
    """把整段文字写进第一个 run，清空其余 run（保留首个 run 的字体）。"""
    runs = p.runs
    if not runs:
        p.add_run(text)
        return True
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""
    return True


def set_para(p, text, strip_math=False):
    if strip_math:
        _strip_math(p)
    return _set_run_text(p, text)


def repl_para(p, old, new):
    """段内替换；跨 run 时仅在段落无公式的情况下合并重写。"""
    if old not in p.text:
        return False
    for r in p.runs:
        if r.text and old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True
    if _has_math(p):
        return "SKIP_MATH"
    _set_run_text(p, p.text.replace(old, new, 1))
    return True


def append_para(p, suffix):
    runs = p.runs
    if runs:
        runs[-1].text = (runs[-1].text or "") + suffix
    else:
        p.add_run(suffix)
    return True


def set_cell(cell, text):
    ps = cell.paragraphs
    for extra in ps[1:]:
        extra._element.getparent().remove(extra._element)
    return set_para(cell.paragraphs[0], text)


def find_row(table, probe):
    """在整行所有单元格中查找 probe。
    优先返回「以 probe 开头」的行，避免 'Scale error' 误命中 'Scale error <= 3% rate'
    反过来被吞掉（表1 的指标名在第 2 列，第 1 列是分组名）。"""
    p = probe.lower().strip()
    loose = None
    for r in table.rows:
        for c in r.cells:
            t = c.text.lower().strip()
            if t.startswith(p):
                return r
            if loose is None and p in t:
                loose = r
    return loose


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PAPER / "0-works-v8.docx"))
    ap.add_argument("--out", default=str(PAPER / "0-works-v9.docx"))
    ap.add_argument("--report", action="store_true", help="只出清单，不写 docx")
    ap.add_argument("--bcd", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()

    m = M(Path(a.bcd))
    log, pend = [], []

    # 0) 报告模式：只解析锚点
    if a.report:
        for idx, keys in ANCHOR_MAP:
            for k in keys:
                label, old, fn = _G[k]
                try:
                    new = fn(m)
                except Exception as e:
                    new = f"<ERR {type(e).__name__}: {e}>"
                mark = "OK" if new else "PENDING"
                if not new:
                    pend.append(f"[{idx}] {label}")
                print(f"{mark:8s} [{idx}] {label:22s} {str(new)[:110]}")
        print(f"\nPENDING: {len(pend)}")
        return

    doc = Document(a.src)
    ps = doc.paragraphs
    if len(ps) <= max(P_CONCL_C, 143):
        raise SystemExit(f"源文档段落数异常：{len(ps)}")

    # 1) 全文静态词组替换
    n_repl = 0
    for old, new in GLOBAL_REPL:
        new_v = new(m) if callable(new) else new
        if not new_v:
            pend.append(f"GLOBAL_REPL 无数据，跳过：{old}")
            continue
        for p in ps:
            r = repl_para(p, old, new_v)
            if r is True:
                n_repl += 1
            elif r == "SKIP_MATH":
                pend.append(f"GLOBAL_REPL 跨run且含公式，跳过：{old}")
    log.append(f"静态词组替换命中 {n_repl} 处")

    # 2) 整段替换
    for idx, mk, strip in STATIC_SET:
        set_para(ps[idx], mk(), strip)
        log.append(f"段落[{idx}] 整段重写 ({len(mk())} 字符)")
    for idx, mk in STATIC_APPEND:
        append_para(ps[idx], mk())
        log.append(f"段落[{idx}] 追加补述")

    # 3) 锚点替换
    for idx, keys in ANCHOR_MAP:
        for k in keys:
            label, old, fn = _G[k]
            try:
                new = fn(m)
            except Exception as e:
                new = None
                pend.append(f"[{idx}] {label} 计算异常 {type(e).__name__}: {e}")
            if not new:
                if not any(x.endswith(label) for x in pend):
                    pend.append(f"[{idx}] {label}")
                continue
            r = repl_para(ps[idx], old, new)
            if r is True:
                log.append(f"段落[{idx}] 锚点 {label} 已更新")
            elif r == "SKIP_MATH":
                pend.append(f"[{idx}] {label} 跨run且含公式，未改")
            else:
                pend.append(f"[{idx}] {label} 原文未命中（v8 措辞可能已变）")

    # 3b) 整段重写（数据驱动；在锚点之后，可覆盖同段锚点结果）
    for idx, fn in REWRITE_SECTIONS:
        try:
            new = fn(m)
        except Exception as e:
            new = None
            pend.append(f"[{idx}] 整段重写计算异常 {type(e).__name__}: {e}")
        if not new:
            if not any(x.startswith(f"[{idx}] 整段重写") for x in pend):
                pend.append(f"[{idx}] 整段重写")
            continue
        set_para(ps[idx], new)
        log.append(f"段落[{idx}] 整段重写 ({len(new)} 字符)")

    # 4) 表格
    for ti, probe, col, fn in TABLE_EDITS:
        if ti >= len(doc.tables):
            pend.append(f"表{ti} 不存在")
            continue
        row = find_row(doc.tables[ti], probe)
        if row is None:
            pend.append(f"表{ti} 未找到行『{probe}』")
            continue
        try:
            new = fn(m)
        except Exception as e:
            pend.append(f"表{ti}『{probe}』列{col} 计算异常 {type(e).__name__}: {e}")
            continue
        if not new:
            pend.append(f"表{ti}『{probe}』列{col} 无数据")
            continue
        set_cell(row.cells[col], str(new))
    log.append(f"表格单元格处理 {len(TABLE_EDITS)} 项")

    # 5) 诚信披露段（插入到 §4.2 之前，最后做以免索引漂移）
    anchor_p = ps[P_DISCLOSE_AT]
    note = anchor_p.insert_paragraph_before(S_41_FIX_HEAD, style=anchor_p.style)
    tail = note.insert_paragraph_before(S_41_FIX_TAIL, style=anchor_p.style)
    log.append(f"在段落[{P_DISCLOSE_AT}]前插入 Implementation revision 两段")
    halo_new = _s_41_halo(m)
    if halo_new:
        anchor_p.insert_paragraph_before(halo_new, style=anchor_p.style)
        log.append(f"在段落[{P_DISCLOSE_AT}]前插入 Implementation revision (boundary harmonisation) 一段")
    else:
        pend.append("boundary harmonisation 披露段无数据（out_downstream_halofix 缺失）")

    # 6) Figure 7（消融 E1 定性对比）插入到 §4.5.1（同样最后做，避免索引漂移）
    if not FIG7_PATH.exists():
        pend.append(f"Figure 7 图片不存在：{FIG7_PATH}")
    else:
        tgt = ps[P_FIG7_BEFORE]
        if not tgt.text.strip().startswith(P_FIG7_PROBE):
            tgt = next((p for p in ps if p.text.strip().startswith(P_FIG7_PROBE)), None)
        if tgt is None:
            pend.append("Figure 7 插入点未找到（§4.5.1 讨论段）")
        else:
            img_p = tgt.insert_paragraph_before(style=tgt.style)
            img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            img_p.paragraph_format.space_after = Pt(0)
            img_p.add_run().add_picture(str(FIG7_PATH), width=Inches(FIG7_WIDTH_IN))
            cap_p = tgt.insert_paragraph_before(style=tgt.style)
            cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_p.paragraph_format.space_after = Pt(0)
            cap_p.add_run("Figure 7. ").bold = True
            cap_p.add_run(S_FIG7_CAP)
            log.append(f"在 §4.5.1 讨论段前插入 Figure 7（{FIG7_WIDTH_IN} in 宽）+ 图题")

    out = Path(a.out)
    doc.save(str(out))
    log.append(f"已写出 {out}")

    rep = [f"# {out.stem} 变更说明", "",
           f"- 源文件：`{a.src}`", f"- 输出：`{out}`", "",
           "## 已执行", ""] + [f"- {x}" for x in log] + ["", "## 待人工确认（PENDING）", ""]
    rep += [f"- {x}" for x in pend] if pend else ["- 无"]
    rep += ["", "## 有意保留未改（不依赖本次重跑）", ""] + [f"- {x}" for x in KEEP_MANUAL]
    (out.parent / (out.stem + "-变更说明.md")).write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[:40]))
    print(f"\nPENDING {len(pend)} 项，详见 {out.stem}-变更说明.md")


# ═══════════════════════════════════════════════════════════════════════
# 表格单元格替换：(table_idx, 行探针, 列号, 生成函数)
# ═══════════════════════════════════════════════════════════════════════
def _pm1(m, key):
    v = _sig(m, key)
    return None if v is None else f"{v * 100:.1f}%"


def _cnt(m, key):
    v = _sig(m, key)
    n = _sig(m, "n")
    return None if (v is None or not n) else int(round(v * n))


def _bris_med(m):
    v = m.d_median("brisque")
    mu = _sig(m, "quality.brisque.mean")
    return None if (v is None or mu is None) else f"{f2(mu)} (median {f2(v)})"


def _tbl_thrpt(m):
    L = m.lat()
    if not L:
        return None
    return (f"{f2(L['mean'])} s/pair mean (median {f2(L['median'])} s; range "
            f"{f2(L['min'])}-{f2(L['max'])} s) on a single 8 GB GPU, serial execution")


def _tbl_e3(m, arm, col):
    v = dig(m.e3(arm), {1: "n", 2: "iou_mean", 3: "pos_err_mean_px", 4: "brisque_mean",
                        5: "brisque_pass_rate"}[col])
    if v is None:
        return None
    if col == 1:
        return str(v)
    if col == 4:
        return f2(v)
    if col == 5:
        return pc(v)
    return f3(v)


def _tbl_e4(m, arm, col):
    v = dig(m.e4(arm), {1: "n", 2: "iou_mean", 3: "pos_err_mean_px", 4: "brisque_mean",
                        5: "brisque_pass_rate"}[col])
    if v is None:
        return None
    if col == 1:
        return str(v)
    if col == 4:
        return f2(v)
    if col == 5:
        return pc(v)
    return f3(v)


def _tbl_bcy(m, arm, col):
    R = m.bcy_rate(arm)
    if col <= 4:
        return None if (not R or len(R) < col or R[col - 1] is None) else f"{R[col - 1]:.3f}"
    t = m.bcy_tot(arm)
    return None if t[0] is None else f"{t[0]} / {t[1]}"


def _tbl_proxy(m, dirname, col):
    key = {1: "n", 2: "R_aerial", 3: "iou_best_mean", 4: "iou_best_median"}[col]
    v = m.ds(dirname, key)
    if v is None:
        return None
    return str(v) if col == 1 else f3(v)


TABLE_EDITS = [
    (0, "End-to-end generation throughput", 1, _tbl_thrpt),
    (1, "Centroid position error", 2,
     lambda m: f"{f3(_sig(m, 'alignment.pos_err_px.mean'))} +/- {f3(_sig(m, 'alignment.pos_err_px.std'))} px"),
    (1, "Centroid position error", 3,
     lambda m: f"[{f3(_sig(m, 'alignment.pos_err_px.min'))}, {f3(_sig(m, 'alignment.pos_err_px.max'))}] px"),
    (1, "Scale error", 2,
     lambda m: f"{f3(_sig(m, 'alignment.scale_err.mean'))} +/- {f3(_sig(m, 'alignment.scale_err.std'))}"),
    (1, "Scale error", 3,
     lambda m: f"[{f3(_sig(m, 'alignment.scale_err.min'))}, {f3(_sig(m, 'alignment.scale_err.max'))}]"),
    (1, "Bounding-box IoU", 2,
     lambda m: f"{f3(_sig(m, 'alignment.iou.mean'))} +/- {f3(_sig(m, 'alignment.iou.std'))}"),
    (1, "Bounding-box IoU", 3,
     lambda m: f"[{f3(_sig(m, 'alignment.iou.min'))}, {f3(_sig(m, 'alignment.iou.max'))}]"),
    (1, "IoU >= 0.5 rate", 2, lambda m: _pm1(m, "alignment.iou_above_0p5_rate")),
    (1, "IoU >= 0.5 rate", 3, lambda m: (lambda c, n: None if c is None else f"{c} / {n} pairs")(_cnt(m, "alignment.iou_above_0p5_rate"), _sig(m, "n"))),
    (1, "Sub-pixel centroid rate", 2, lambda m: _pm1(m, "alignment.sub_pixel_rate")),
    (1, "Sub-pixel centroid rate", 3, lambda m: (lambda c, n: None if c is None else f"{c} / {n} pairs")(_cnt(m, "alignment.sub_pixel_rate"), _sig(m, "n"))),
    (1, "Scale error <= 3% rate", 2, lambda m: _pm1(m, "alignment.scale_below_3pct_rate")),
    (1, "Scale error <= 3% rate", 3, lambda m: (lambda c, n: None if c is None else f"{c} / {n} pairs")(_cnt(m, "alignment.scale_below_3pct_rate"), _sig(m, "n"))),
    (1, "Cross-modal IoU", 2, lambda m: "1.000 (identical by construction)"),
    (1, "Cross-modal IoU", 3, lambda m: (lambda n: None if not n else f"{n} / {n} (not an independent measurement)")(_sig(m, "n"))),
    (1, "BRISQUE score", 2, _bris_med),
    (1, "BRISQUE score", 3, lambda m: f"[{f2(_sig(m, 'quality.brisque.min'))}, {f2(_sig(m, 'quality.brisque.max'))}]"),
    (1, "Quality pass rate", 2, lambda m: _pm1(m, "quality.pass_rate")),
    (1, "Quality pass rate", 3, lambda m: (lambda c, n: None if c is None else f"{c} / {n} accepted")(_cnt(m, "quality.pass_rate"), _sig(m, "n"))),
    (1, "Horizontal coordinate entropy", 2, lambda m: f"{f3(_sig(m, 'diversity.entropy_nu_bits'))} bits"),
    (1, "Vertical coordinate entropy", 2, lambda m: f"{f3(_sig(m, 'diversity.entropy_nv_bits'))} bits"),
    (1, "Target scale entropy", 2, lambda m: f"{f3(_sig(m, 'diversity.entropy_scale_bits'))} bits"),
    (1, "Unique condition tuples", 2,
     lambda m: f"{_sig(m, 'diversity.distinct_tuples')} / {_sig(m, 'diversity.total_tuples')}"),
    (2, "LCR", 1, lambda m: f3(m.fus("out_E1_full", "lcr"))),
    (2, "LCR", 2, lambda m: f3(m.fus("out_E1_paste", "lcr"))),
    (2, "LCR", 3, lambda m: (lambda a, b: None if a is None else f"{a - b:+.3f}")(m.fus("out_E1_full", "lcr"), m.fus("out_E1_paste", "lcr"))),
    (2, "LCR", 4, lambda m: (lambda p: None if p is None else f"{p:.1e}")(m.fus_p("lcr"))),
    (2, "RLJ", 1, lambda m: f3(m.fus("out_E1_full", "rlj"))),
    (2, "RLJ", 2, lambda m: f3(m.fus("out_E1_paste", "rlj"))),
    (2, "RLJ", 3, lambda m: (lambda a, b: None if a is None else f"{a - b:+.3f}")(m.fus("out_E1_full", "rlj"), m.fus("out_E1_paste", "rlj"))),
    (2, "RLJ", 4, lambda m: (lambda p: None if p is None else f"{p:.1e}")(m.fus_p("rlj"))),
]

# 按 (表, 行, 臂, 列) 批量生成单元格编辑项
def _mk(gen, idx, probe, arm, cols):
    for c in cols:
        TABLE_EDITS.append((idx, probe, c, (lambda c=c: lambda m: gen(m, arm, c))()))


_mk(_tbl_e3, 3, "E3 LoRA merged", "lora", (1, 2, 3, 4, 5))
_mk(_tbl_e3, 3, "E3 LoRA disabled", "nolora", (1, 2, 3, 4, 5))
_mk(_tbl_e4, 3, "E4 depth + segmentation", "both", (1, 2, 3, 4, 5))
_mk(_tbl_e4, 3, "E4 depth only", "depth", (1, 2, 3, 4, 5))
_mk(_tbl_e4, 3, "E4 segmentation only", "seg", (1, 2, 3, 4, 5))
_mk(_tbl_bcy, 4, "A1 open loop", "A1", (1, 2, 3, 4, 5))
_mk(_tbl_bcy, 4, "A2 rejection resampling", "A2", (1, 2, 3, 4, 5))
_mk(_tbl_bcy, 4, "A3 full loop", "A3", (1, 2, 3, 4, 5))
_mk(_tbl_proxy, 5, "Primary benchmark", "out_D", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E1 repainting", "out_E1_full", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E1 hard paste", "out_E1_paste", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E3 LoRA merged", "out_E3_lora", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E3 LoRA disabled", "out_E3_nolora", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E4 depth + segmentation", "out_E4_both", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E4 depth only", "out_E4_depth", (1, 2, 3, 4))
_mk(_tbl_proxy, 5, "E4 segmentation only", "out_E4_seg", (1, 2, 3, 4))
TABLE_EDITS.append((5, "Background-only control", 2,
                    lambda m: f3(dig(m.ds("out_D", "fpr"), "fp_aerial_rate"))))


# 静态段落改写：(段落 idx, 文本生成, 是否清除段内公式)
STATIC_SET = [
    (74, lambda: S_M5_BODY, False),
    (78, lambda: S_M5_BLEND, False),
    (80, lambda: S_M5_WHERE, True),
    (82, lambda: S_FRRM, False),
]
def _m5_cfg_tail():
    t = S_M5_CFG.split(". ", 1)[1]
    return " " + t[0].lower() + t[1:]


STATIC_APPEND = [(76, _m5_cfg_tail)]


def _m_fig7_ref():
    return S_FIG7_REF


if FIG7_PATH.exists():
    STATIC_APPEND.append((P_E1_INTRO, _m_fig7_ref))



if __name__ == "__main__":
    main()
