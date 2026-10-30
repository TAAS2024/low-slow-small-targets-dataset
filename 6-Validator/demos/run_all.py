#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  6-Validator 全链路 Demo Runner
═══════════════════════════════════════════════════════════════
依次运行 S1-S5 所有 Demo，展示输入端预校验全流程。

用法:
  python3 demos/run_all.py          # 运行全部
  python3 demos/run_all.py --s1     # 只运行 S1
  python3 demos/run_all.py --quick  # 快速模式（减少场景）
"""
import os, sys, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMOS = {
    "S1": "S1 Agent 1 JSON校验/demos/demo_s1.py",
    "S2": "S2 Agent 2 Transformer校验/demos/demo_s2.py",
    "S3": "S3 Agent 3 ControlNet校验/demos/demo_s3.py",
    "S4": "S4 Agent 4 LoRA校验/demos/demo_s4.py",
    "S5": "S5 Agent 5 IR校验/demos/demo_s5.py",
}

DESCRIPTIONS = {
    "S1": "Agent 1 (LLM JSON) — 6 项：Schema/枚举/逻辑/描述/轨迹/相机",
    "S2": "Agent 2 (Transformer) — 6 项：越界/尺寸/位置/seg/depth/帧数",
    "S3": "Agent 3 (ControlNet) — 4 项：尺寸/位置/边界/depth一致性",
    "S4": "Agent 4 (LoRA) — 3 项：概念泄漏/纹理重复/色偏",
    "S5": "Agent 5 (IR) — 3 项：尺寸/灰度/直方图 (只告警不截停)",
}


def run_demo(key):
    path = os.path.join(ROOT, DEMOS[key])
    print(f"\n{'#'*64}")
    print(f"#  {key} — {DESCRIPTIONS[key]}")
    print(f"{'#'*64}")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print(f"  [stderr]: {result.stderr.strip()}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="S1-S5 Validator 全链路 Demo")
    parser.add_argument("--s1", action="store_true")
    parser.add_argument("--s2", action="store_true")
    parser.add_argument("--s3", action="store_true")
    parser.add_argument("--s4", action="store_true")
    parser.add_argument("--s5", action="store_true")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    args = parser.parse_args()

    selected = [k for k in DEMOS if getattr(args, k.lower(), False)]
    if not selected:
        selected = list(DEMOS.keys())

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                            ║
║     6-Validator 输入端预校验 — S1→S5 全链路 Demo           ║
║                                                            ║
║  管线: S1(JSON) → S2(Transformer) → S3(ControlNet)         ║
║        → S4(LoRA) → S5(IR) → [生成管线] → S6-S9(输出审查)  ║
║                                                            ║
╚══════════════════════════════════════════════════════════════╝
""")

    passed = 0
    for key in selected:
        if run_demo(key):
            passed += 1

    print(f"\n{'#'*64}")
    print(f"#  🏁 全链路 Demo 完成: {passed}/{len(selected)} 个 Validator 运行成功")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()
