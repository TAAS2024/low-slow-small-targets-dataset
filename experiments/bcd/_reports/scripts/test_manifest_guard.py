#!/usr/bin/env python3
"""manifest 写保护单元测试（离线，无需 GPU）。

复现并验证 HALO-FIX-2026-10-06.md §4.6-A 的缺陷：`ablation_driver.gen()`
对**每个 item 单独**调用 `generate_batch([it], flush_every=1)`，旧实现
每次整体覆写 manifest → 磁盘上只剩最后 1 条。

运行：
    python3 _reports/scripts/test_manifest_guard.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BCD = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/experiments/bcd")
sys.path.insert(0, str(BCD))

import harness  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILED.append(name)


def rec(sid: str, skip_cnet: bool = False) -> dict:
    return {"sample_id": sid, "rgb": f"samples/{sid}/final_rgb.png",
            "bbox": [1.0, 2.0, 3.0, 4.0], "skip_cnet": skip_cnet}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mfguard_"))
    try:
        # ── 场景 1：逐条写入（完全复刻 gen() 的调用模式）──
        print("\n[场景 1] 逐条调用 → manifest 应累积而非被覆写")
        for sid in ("d0000", "d0001", "d0002"):
            n = harness.write_manifest(tmp, [rec(sid)])
        man = json.loads((tmp / "manifest.json").read_text())
        sids = [s["sample_id"] for s in man["samples"]]
        check("3 条全部保留", sids == ["d0000", "d0001", "d0002"], str(sids))
        check("收尾计数正确", n == 3, f"n={n}")

        # ── 场景 2：重复写入同一 sample_id → 去重且新值优先 ──
        print("\n[场景 2] 同一 sample_id 重复写入 → 去重，新值胜出")
        r2 = rec("d0001")
        r2["bbox"] = [9.0, 9.0, 9.0, 9.0]
        harness.write_manifest(tmp, [r2])
        man = json.loads((tmp / "manifest.json").read_text())
        sids = [s["sample_id"] for s in man["samples"]]
        row = next(s for s in man["samples"] if s["sample_id"] == "d0001")
        check("仍为 3 条（去重）", len(sids) == 3, str(sids))
        check("新值覆盖旧值", row["bbox"] == [9.0, 9.0, 9.0, 9.0], str(row["bbox"]))
        check("顺序稳定", sids == ["d0000", "d0001", "d0002"], str(sids))

        # ── 场景 3：原子写 → 无 .tmp 残留 ──
        print("\n[场景 3] 原子写 → 目录内不应残留 .tmp*")
        leftovers = [p.name for p in tmp.iterdir() if ".tmp" in p.name]
        check("无临时文件残留", not leftovers, str(leftovers))

        # ── 场景 4：半截 JSON → 能自愈，以本次为准 ──
        print("\n[场景 4] 磁盘上是被截断的 JSON → 自愈（丢弃损坏内容）")
        (tmp / "manifest.json").write_text('{"samples": [{"sample_id": "d0000"',
                                           encoding="utf-8")
        n = harness.write_manifest(tmp, [rec("d9999")])
        man = json.loads((tmp / "manifest.json").read_text())
        sids = [s["sample_id"] for s in man["samples"]]
        check("损坏文件被替换", sids == ["d9999"], str(sids))

        # ── 场景 5：skip_cnet 不一致 → 不混入其它臂 ──
        print("\n[场景 5] 目录复用时 skip_cnet 不同 → 丢弃历史条目")
        harness.write_manifest(tmp, [rec("e0000")], skip_cnet=False)
        harness.write_manifest(tmp, [rec("e0001")], skip_cnet=True)
        man = json.loads((tmp / "manifest.json").read_text())
        sids = [s["sample_id"] for s in man["samples"]]
        check("skip_cnet 切换后只留新臂", sids == ["e0001"], str(sids))

        # ── 场景 6：_merge 保持原顺序 ──
        print("\n[场景 6] 合并顺序：历史在前，新增追加在后")
        d = tmp / "sub"
        d.mkdir()
        for sid in ("a3", "a1", "a2"):
            harness.write_manifest(d, [rec(sid)])
        sids = [s["sample_id"] for s in
                json.loads((d / "manifest.json").read_text())["samples"]]
        check("顺序 = 写入顺序", sids == ["a3", "a1", "a2"], str(sids))

        # ── 场景 7：eval_downstream.load_samples 守护告警路径 ──
        print("\n[场景 7] eval_downstream.load_samples 一致性守护")
        import eval_downstream as ed

        def mk(name: str, declared: int, on_disk: int) -> Path:
            """造一个：manifest 声明 declared 条，磁盘只放 on_disk 个 PNG。"""
            d = tmp / name
            for i in range(on_disk):
                (d / "samples" / f"s{i}").mkdir(parents=True, exist_ok=True)
                (d / "samples" / f"s{i}" / "final_rgb.png").write_bytes(b"\x89PNG")
            (d / "manifest.json").write_text(json.dumps({"samples": [
                {"sample_id": f"s{i}", "rgb": f"samples/s{i}/final_rgb.png",
                 "bbox": [0, 0, 1, 1]} for i in range(declared)]}), encoding="utf-8")
            return d

        # 7a. 几乎全丢（历史事故形态）
        got = ed.load_samples(mk("g_total", 2, 1), None)
        check("7a 声明 2/可用 1 → 告警且只返回 1 条", len(got) == 1, f"n={len(got)}")

        # 7b. 部分截断（旧版 `len(out) <= 1` 漏掉的情形）
        got = ed.load_samples(mk("g_part", 5, 2), None)
        check("7b 声明 5/可用 2 → 应告警（部分截断）", len(got) == 2, f"n={len(got)}")

        # 7c. 显式 --limit 属主动裁剪，不得误报
        got = ed.load_samples(mk("g_lim", 5, 5), 2)
        check("7c 声明 5/可用 5 + limit 2 → 不误报", len(got) == 2, f"n={len(got)}")

        # 7d. 声明完好、磁盘齐全 → 静默
        got = ed.load_samples(mk("g_ok", 3, 3), None)
        check("7d 声明 3/可用 3 → 静默通过", len(got) == 3, f"n={len(got)}")

        print("\n" + "=" * 56)
        if FAILED:
            print(f"❌ {len(FAILED)} 项失败: {FAILED}")
            return 1
        print("✅ 全部通过（7 场景）")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
