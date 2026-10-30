"""
Run New Demo — Transformer B + ControlNet 端到端生成
=====================================================
替代旧的 run_demo_768.py，使用训练好的 Transformer B 模型生成
depth+seg 条件图，再送入 ControlNet 生成最终场景。

用法:
    python run_demo_new.py --model ../4-Transformer/output/layer2/layer2_best.pt
    python run_demo_new.py --model ../4-Transformer/output/layer2/layer2_best.pt --scenes demo_scenes.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformer2controlnet import TransformerToControlNet


def main():
    parser = argparse.ArgumentParser(description="New Demo: Transformer B → ControlNet")
    parser.add_argument("--model", required=True, help="Trained Transformer B checkpoint (.pt)")
    parser.add_argument("--scenes", default="demo_scenes.json", help="Scene definitions JSON")
    parser.add_argument("--output", default="outputs/demo_new", help="Output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    # 加载场景定义
    scenes_path = Path(args.scenes)
    if not scenes_path.exists():
        scenes_path = Path(__file__).parent / args.scenes
    
    with open(scenes_path) as f:
        demo_data = json.load(f)
    
    print(f"Demo: {demo_data.get('demo_name', 'Unnamed')}")
    print(f"Description: {demo_data.get('description', '')}")
    print(f"Scenes: {len(demo_data['scenes'])}")
    
    # 初始化桥接
    bridge = TransformerToControlNet(args.model, device=args.device)
    
    # 批量生成
    out_dir = bridge.generate_demo_set(demo_data["scenes"], args.output)
    
    # 保存 demo 元数据
    with open(Path(out_dir) / "demo_info.json", "w") as f:
        json.dump({
            "demo_name": demo_data.get("demo_name", ""),
            "description": demo_data.get("description", ""),
            "model": args.model,
            "pipeline": "Agent1 JSON → Transformer B (depth+seg) → Dual ControlNet → RGB Scene",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Demo complete! Results: {out_dir}/")
    for scene in demo_data["scenes"]:
        print(f"  {scene['label']}/")
        print(f"    ├── depth.png          — Transformer B 输出深度图")
        print(f"    ├── seg.png            — Transformer B 输出分割图")
        print(f"    ├── step1_background.png — ControlNet 背景生成")
        print(f"    ├── step2_with_drone.png — 无人机合成")
        print(f"    └── step3_final.png    — 最终融合图像")


if __name__ == "__main__":
    main()
