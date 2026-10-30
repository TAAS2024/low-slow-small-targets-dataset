# `_reports/` — v10 阶段的产物与存档

> 2026-10-07 建立。原先散落在 Windows 桌面的状态文件、对比图、一次性脚本，统一收拢到实验目录内，
> 避免污染桌面、也避免与项目脱节。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `v10_status.txt` | v10 全链执行状态与最终结论（人类可读流水账，含 4 个评测链路 bug 的定位） |
| `v10_stage.txt` | 单行阶段标记（脚本每步覆写，供快速查看当前阶段） |
| `halo_abc_compare.png` | **A/B/C 三组视觉对照**（6 样本 × 3 组，LANCZOS ×5 放大，4576×5771）。由 `visual_abc_compare.py` 生成 |
| `halo_fix_compare.png` | 融合带修复前/后对照（最脏 4 例，含 \|diff\|×4 栏）。由 `visual_halo_fix.py` 生成 |
| `halo_check.png` | 指定样本的 bg / final_rgb / 差分三栏图。由 `visual_halo.py` 生成 |
| `smoke_guard_2026-10-07.log` | 16 样本端到端 smoke 的关键日志（逐条 `manifest 累计 N` 递增痕迹、收尾补写、VRAM） |
| `smoke_endpoints_2026-10-11.json` | **Web 端点冒烟报告**（33/33 通过）。由 `0-workspace/app/smoke_endpoints.py` 生成，无需 GPU |
| `exclude_queue_probe_2026-10-11.json` | **滑动排除队列行为探针**：4 组 (scene/tod/weather) × 30 次检索的完整取图序列，验证论文 §3.3.1 的窗口语义。由 `0-workspace/app/probe_exclude_queue.py` 生成，无需 GPU |
| `provenance_check/conf025/` | **2026-10-11 数据溯源复跑**：对 `_v9_prehalofix/out_D`、`out_D`、`out_D_halofix` 三者以官方口径（`--conf 0.25 --batch 8 --fpr`）重跑 YOLO，确认 0.650/0.5567、0.6167/0.5067、0.6200/0.5200 全部逐位复现 |
| `guard_evidence/` | 写保护三测的产物快照：`downstream.OK.json`（n 匹配，rc=0）/ `downstream.GUARD_FAILED.json`（n 不符，rc=2，未覆盖正式产物）/ `N_GUARD_FAILED.txt` |

## 复现下游代理的两个坑（2026-10-11 实测确认）

1. **必须显式传 `--conf 0.25`**。`eval_downstream.py` 的参数默认值是 `0.05`（"低阈值以观察弱信号"），
   用它跑出来的是另一套数（`out_D` → R_any 0.833 / R_aerial 0.667），与论文表 6 完全不可比。
   官方口径见 `run_v9.sh`：`--conf 0.25 --batch 8 --fpr --expect-n ...`。
2. **`out_downstream_halofix/downstream.json` 的 `out_D` 臂是历史标签**。该臂记录的是
   **修正前**渲染集的评测结果，但 `out_dir` 字段写的仍是 `.../out_D`；该路径已于
   2026-10-07 07:35 被 v10 全链重跑覆写为**修正后**渲染。修正前的 300 张图像现存于
   `_v9_prehalofix/out_D/samples`。两个 JSON 内的 `out_dir_note` 字段已就地说明。
   复跑本臂须指向归档路径，否则会得到 0.617/0.507 而非 0.650/0.557。

## `repair_manifests.py` 的定位（2026-10-11 明确）

**它只是兜底工具，不是流水线的一环。** 历史上 manifest 会被「跑一半被打断」的进程写成半截 JSON、
或在增量追加时丢失既有条目，`repair_manifests.py` 就是为修复这类**历史目录**而写的一次性补丁。

现在源头已经堵住：`harness.py` / `eval_downstream.py` 的 manifest 写入走
`_merge_manifest_records()`（读旧条 → 按 key 累积合并 → 去重保序）+ `_atomic_write_json()`
（先写 `.tmp` 再 `os.replace`），并带 `--expect-n` 守护（样本数不符即 rc=2 且拒绝覆盖正式产物）；
离线单测见 `scripts/test_manifest_guard.py`（10 断言，秒级）。

因此：**新产物一律不需要跑 `repair_manifests.py`**。只有在遇到旧目录（v9 及更早）缺条、
或人为中断留下的半截文件时才用它，用完即弃，不要把它加进任何自动化脚本。

## `scripts/` — 一次性脚本存档

| 脚本 | 用途 | 状态 |
| --- | --- | --- |
| `run_stage2.sh` | STAGE2 编排：等 A/B 结束 → 跑 C(HARD) → 自动择优 → 接 v10 全链 | 已跑完，可复用 |
| `rerun_v10.sh` | halo 修复后的全链重跑（留档 v9 → smoke → core → loop → manifest 修复） | 已跑完，可复用 |
| `smoke_ab.sh` | A/B 对照 smoke（16 样本，before/after 各一遍 + 诊断） | 可复用 |
| `sigma_scan.sh` | σ 参数扫描 | 可复用 |
| `v10_halo_env.sh` | 择优后的环境变量落盘（`BCD_HALO_FIX` / `BCD_HALO_SIGMA`） | 当前值：SOFT，σ=1.0 |
| `analyze_abc_paired.py` | A/B/C 逐样本配对统计（胜率、p10/p90、逐样本 best） | 可复用 |
| `check_paper_vals.py` | 论文 docx 旧值残留 / 新值确认扫描（22 项旧值 + 20 项新值） | 可复用 |
| `v10_final_check.py` | v9 vs v10 指标、下游、docx 数字出现次数三向对照 | 可复用 |
| `test_manifest_guard.py` | **manifest 写保护单元测试**（离线，7 场景 / 10 断言：累积合并 / 去重顺序 / 无 tmp 残留 / 半截 JSON 自愈 / skip_cnet 隔离 / 合并顺序 / 读方一致性守护 4 项） | 每次改动 `harness.py` 或 `eval_downstream.py` 后必跑 |
| `verify_v12_structure.py` | **论文结构/措辞校验**（33 断言）：段表数、披露段链路、无未实现声明、排除队列窗口语义、显著性口径、表 6 控制行、数字卫生、包完整性 | 每次重生成 docx 后必跑 |
| `recompute_from_raw.py` | **数据溯源复算**（80 断言）：从 `out_D_eval/results.json` 与 `out_downstream/downstream.json` 的**逐样本记录**重算 §4.3 全部统计量、表 6 全部单元格、配对 McNemar、halo 三方溯源 | 方法学变更或产物更新后必跑 |

## 复现命令

```bash
cd /mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/experiments/bcd
python3 visual_abc_compare.py        # → _reports/halo_abc_compare.png
python3 visual_halo_fix.py           # → _reports/halo_fix_compare.png
python3 _reports/scripts/check_paper_vals.py   # 论文数字质检
python3 _reports/scripts/verify_v12_structure.py      # 论文结构/措辞校验（33 项）
python3 _reports/scripts/recompute_from_raw.py        # 数据溯源复算：从原始逐样本记录重算 §4.3/表6/halo（80 项）
python3 _reports/scripts/test_manifest_guard.py  # 写保护单测（无需 GPU，秒级）
cd ../../0-workspace/app && python3 smoke_endpoints.py \
  --json "../../experiments/bcd/_reports/smoke_endpoints_$(date +%F).json"   # Web 端点冒烟（33 项，无需 GPU）
cd ../../0-workspace/app && python3 probe_exclude_queue.py \
  --json "../../experiments/bcd/_reports/exclude_queue_probe_$(date +%F).json"  # 排除队列行为探针（无需 GPU）

# 写保护的端到端 smoke（需 GPU，16 样本约 5 min）
python3 -u ablation_driver.py --exp E1 --cond full --n 16 --out _smoke_guard/out_smoke_guard
# 判据：日志出现 `manifest 累计 1…16`，且 out_smoke_guard/manifest.json 有 16 条
```

## 约定

- **本目录只存归档产物**，不参与实验流水线；生成脚本仍放在 `experiments/bcd/` 根目录。
- 状态文件与对比图**一律输出到 `_reports/`**，不再写 Windows 桌面。
- 日志仍走 `/tmp/`（运行期临时文件，任务结束即弃）。
