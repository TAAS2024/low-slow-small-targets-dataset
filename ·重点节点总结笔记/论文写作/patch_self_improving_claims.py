#!/usr/bin/env python3
"""按 v6 清单第 5 条规则，撤除与 B 组实测不符的「自改进」类断言。

规则：`"self-improving"` 仅当 B 组曲线**确实单调上升**时方可保留。
实测（out_B_A{1,2,3}/curve.json，2026-09-29 完成）：
  A1 接受率  0.150 → 0.233 → 0.217 → 0.217   （首轮跳升后持平，非单调）
  A2 接受率  0.144 → 0.117 → 0.144 → 0.139   （无趋势）
  A3 接受率  0.144 → 0.172 → 0.185 → 0.176   （前三轮升，末轮回落）
→ 触发规则：删除对本框架的 self-improving / co-evolve / iterative
  quality bootstrapping 断言，改为与实测一致的中性表述。

保留：§1.1 中「现有管线无法自我改进」属**对既有工作的局限陈述**，
v4 本身即如此表述，不在本次修改范围。

幂等：可重复执行；已替换过则报 0 处改动。
"""

from __future__ import annotations

from pathlib import Path

FILES = [
    Path("0-works-v3-审校稿.md"),
    Path("论文写作/0-works-v3-审校稿.md"),
]

EN_LOOP_OLD = (
    "Furthermore, the framework incorporates a closed-loop adversarial feedback "
    "mechanism, in which the generator and validator co-evolve under physical-anchor "
    "constraints, with curriculum hardening that escalates validation thresholds, "
    "recalibrates the validator against hard negatives, and penalises mode collapse, "
    "thereby enabling iterative quality bootstrapping without requiring continuous "
    "human intervention."
)
EN_LOOP_NEW = (
    "Furthermore, the framework incorporates a closed-loop adversarial feedback "
    "mechanism operating under physical-anchor constraints, with curriculum hardening "
    "that escalates validation thresholds and recalibrates the validator against hard "
    "negatives, so that accepted samples are retained while each rejection is routed "
    "back to the validator stage that produced it."
)

CN_LOOP_OLD = (
    "此外，该框架整合了闭环对抗反馈机制，生成器与验证器在物理锚定约束下协同进化，"
    "配合课程硬化逐步提高验证阈值、依据难负样本重校准验证器、并惩罚模式坍缩，"
    "从而无需持续人工干预即可实现迭代式质量自举。"
)
CN_LOOP_NEW = (
    "此外，该框架整合了物理锚定约束下的闭环对抗反馈机制，配合课程硬化逐步提高验证阈值、"
    "并依据难负样本重校准验证器；被接受的样本予以保留，而每一次拒绝都被回送到产生它的验证阶段。"
)

EN_RES_OLD = (
    "Experiments demonstrate that the closed-loop framework progressively refines "
    "generated samples, advancing from initial outputs that exhibit background conflicts "
    "and structural distortions toward background-consistent targets with stable "
    "structure, confirmed flight trajectories, and verified realism together with accurate "
    "bounding box annotations, while ablation studies confirm the individual contributions "
    "of marker injection, multi-stage validation, closed-loop feedback, and curriculum "
    "hardening. By synthesising training data for LSS categories that cannot be collected "
    "in practice, the framework establishes a reproducible and scalable paradigm that "
    "generalises beyond the specific detection architecture employed, thereby providing a "
    "principled basis for controllable, verifiable, and self-improving synthetic data "
    "generation for aerial target detection."
)
EN_RES_NEW = (
    "Experiments characterise this loop empirically rather than claim monotonic "
    "self-improvement. Across four closed-loop cycles the per-attempt acceptance rate "
    "rises from 0.150 to a plateau of 0.217-0.233 and thereafter remains flat; retry-based "
    "rejection and curriculum hardening instead trade per-attempt yield for per-target "
    "completion, raising completion from 0.150 to 0.433 while lowering yield from 0.204 to "
    "0.136-0.167. Ablation studies confirm the individual contributions of marker "
    "injection, multi-stage validation, and closed-loop feedback. Because the terminal "
    "validator stage is a COCO-pretrained YOLOv8x that has no drone class, the loop's "
    "rejection signal measures detectability relative to that fixed detector; the "
    "resulting data must therefore not be used to evaluate the recall of YOLOv8x-family "
    "models, and the loop is not claimed to improve generation quality in general. By "
    "synthesising training data for LSS categories that cannot be collected in practice, "
    "the framework establishes a reproducible and scalable paradigm that generalises "
    "beyond the specific detection architecture employed, while its downstream utility "
    "remains to be established on held-out real data."
)

CN_RES_OLD = (
    "实验表明，闭环框架逐步精炼生成样本，从初始输出中存在的背景冲突与结构失真，"
    "推进到背景一致、结构稳定、飞行轨迹得到确认、真实感得到验证且边界框标注准确的目标；"
    "消融实验则确认了标记注入、多阶段验证、闭环反馈与课程硬化各自的贡献。"
    "通过为实际无法采集的 LSS 类别合成训练数据，该框架建立了一种可复现、可扩展的范式，"
    "能够泛化到所采用的具体检测架构之外，"
    "从而为面向空中目标检测的可控、可验证、自改进的合成数据生成提供了原则性基础。"
)
CN_RES_NEW = (
    "实验对这一回路作经验刻画，而不主张单调的自我改进。在四个闭环循环中，"
    "每次尝试的接受率由 0.150 升至 0.217–0.233 的平台后即持平；"
    "基于重试的拒绝采样与课程硬化则以单位尝试产出率换取目标达成率——"
    "目标达成率由 0.150 提升至 0.433，产出率则由 0.204 降至 0.136–0.167。"
    "消融实验确认了标记注入、多阶段验证与闭环反馈各自的贡献。"
    "由于验证器末级为 COCO 预训练、不含 drone 类的 YOLOv8x，"
    "闭环的拒绝信号度量的是相对该既定检测器的可检出性；"
    "因此所得数据不得用于评测 YOLOv8x 同族模型的召回率，亦不主张该回路普遍提升了生成质量。"
    "通过为实际无法采集的 LSS 类别合成训练数据，该框架建立了一种可复现、可扩展的范式，"
    "能够泛化到所采用的具体检测架构之外，其下游效用仍须在留出的真实数据上加以确立。"
)

PATCHES = [
    ("§1 摘要/方法段 EN：co-evolve + iterative quality bootstrapping", EN_LOOP_OLD, EN_LOOP_NEW),
    ("§1 摘要/方法段 CN：协同进化 + 迭代式质量自举", CN_LOOP_OLD, CN_LOOP_NEW),
    ("§1 摘要/结论段 EN：progressively refines + self-improving", EN_RES_OLD, EN_RES_NEW),
    ("§1 摘要/结论段 CN：逐步精炼 + 自改进", CN_RES_OLD, CN_RES_NEW),
    ("§1.4 贡献 4 EN：prevent mode collapse + sustained quality bootstrapping",
     "which together prevent reward hacking and mode collapse and enable sustained "
     "quality bootstrapping across successive generation cycles.",
     "which together mitigate reward hacking and ensure that every rejection can be "
     "routed back to a specific generator module. We do not claim that this loop yields "
     "monotonic quality improvement; its measured behaviour is reported in Section 4.5.7."),
    ("§1.4 贡献 4 CN：防止模式坍缩 + 持续的质量自举",
     "它们共同防止奖励欺骗与模式坍缩，并在连续生成周期中实现持续的质量自举。",
     "它们共同抑制奖励欺骗，并使每一次拒绝都能回流到具体的生成器模块。"
     "本文不主张该回路带来单调的质量提升，其实际行为由第 4.5.7 节的闭环实测刻画。"),
]


def main() -> None:
    for f in FILES:
        if not f.exists():
            print(f"[skip] {f} 不存在")
            continue
        txt = f.read_text(encoding="utf-8")
        changed = 0
        for label, old, new in PATCHES:
            if old in txt:
                txt = txt.replace(old, new, 1)
                changed += 1
                print(f"  [改] {label}")
            elif new in txt:
                print(f"  [已改] {label}")
            else:
                print(f"  [!! 未匹配] {label}")
        if changed:
            f.write_text(txt, encoding="utf-8")
        print(f"{f}: 本次替换 {changed} 处\n")


if __name__ == "__main__":
    main()
