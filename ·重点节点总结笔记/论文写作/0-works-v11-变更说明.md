# 0-works-v11 变更说明

- 源文件：`/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/·重点节点总结笔记/论文写作/0-works-v8.docx`
- 输出：`/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/·重点节点总结笔记/论文写作/0-works-v11.docx`

## 已执行

- 静态词组替换命中 11 处
- 段落[74] 整段重写 (517 字符)
- 段落[78] 整段重写 (169 字符)
- 段落[80] 整段重写 (713 字符)
- 段落[82] 整段重写 (1203 字符)
- 段落[76] 追加补述
- 段落[167] 追加补述
- 段落[3] 锚点 abs_pos 已更新
- 段落[3] 锚点 abs_scale 已更新
- 段落[3] 锚点 abs_proxy 已更新
- 段落[3] 锚点 abs_loop 已更新
- 段落[3] 锚点 abs_cost 已更新
- 段落[3] 锚点 abs_tradeoff 已更新
- 段落[3] 锚点 abs_raise 已更新
- 段落[125] 锚点 p41_usecfg 已更新
- 段落[125] 锚点 p41_lat 已更新
- 段落[125] 锚点 p41_lat_tod 已更新
- 段落[143] 锚点 p43_pos 已更新
- 段落[143] 锚点 p43_iou 已更新
- 段落[143] 锚点 p43_scale 已更新
- 段落[143] 锚点 p43_rates 已更新
- 段落[143] 锚点 p43_split 已更新
- 段落[143] 锚点 p43_resid 已更新
- 段落[147] 锚点 p43_tod 已更新
- 段落[148] 锚点 p43_bris_abs 已更新
- 段落[148] 锚点 p43_bris_scene 已更新
- 段落[148] 锚点 p43_entropy 已更新
- 段落[170] 锚点 e1_geom 已更新
- 段落[171] 锚点 e1_lcr 已更新
- 段落[171] 锚点 e1_bge 已更新
- 段落[172] 锚点 e1_proxy 已更新
- 段落[177] 锚点 e3_geom 已更新
- 段落[177] 锚点 e3_proxy 已更新
- 段落[179] 锚点 e4_geom 已更新
- 段落[179] 锚点 e4_quality 已更新
- 段落[179] 锚点 e4_bge 已更新
- 段落[179] 锚点 e4_lcr 已更新
- 段落[179] 锚点 e4_rlj 已更新
- 段落[179] 锚点 e4_proxy 已更新
- 段落[179] 锚点 e4_bge_frame 已更新
- 段落[179] 锚点 e4_rlj_frame 已更新
- 段落[179] 锚点 e4_proxy_frame 已更新
- 段落[182] 锚点 e5_cos 已更新
- 段落[184] 锚点 e6_curve 已更新
- 段落[184] 锚点 e6_marginal 已更新
- 段落[186] 锚点 b_a1_curve 已更新
- 段落[186] 锚点 b_a2_gain 已更新
- 段落[186] 锚点 b_a3_gain 已更新
- 段落[188] 锚点 b_caveat 已更新
- 段落[188] 锚点 b_caveat_frame 已更新
- 段落[210] 锚点 concl_fourth 已更新
- 段落[172] 整段重写 (1208 字符)
- 段落[190] 整段重写 (1602 字符)
- 段落[192] 整段重写 (708 字符)
- 表格单元格处理 104 项
- 在段落[127]前插入 Implementation revision 两段
- 在 §4.5.1 讨论段前插入 Figure 7（4.8 in 宽）+ 图题
- 已写出 /mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/·重点节点总结笔记/论文写作/0-works-v11.docx

## 待人工确认（PENDING）

- 无

## 有意保留未改（不依赖本次重跑）

- lower the false-acceptance rate by 79.0% relative to a single perceptual threshold
- field-level accuracy of 0.985 and an exact schema match rate of 0.93
- peak footprint of about 10 GB
- lowering the peak to 4.55 GB

---

# 本次 v10 → v11 的五项定向修复（2026-10-09）

数据源：`experiments/bcd/out_D_eval/results.json`（300 对，逐样本含 W/H 与 alignment）
画布实测：中位分辨率 **1280 × 853**，中位对角 **1538.18 px**（均值对角 1753.84 px）

## ① 画布对角口径统一 → 0.080%

| 位置 | v10 | v11 | 依据 |
|---|---|---|---|
| 摘要 | `0.080%`（动态） | `0.080%` | 1.226 / 1538.18 = 0.0797% |
| §4.3 | `0.072%`（硬编码） | **`0.080%`** | 同上；反推 0.072% 需对角 1702.8 px，属上一轮画布家族残留 |
| §5 结论 | `0.072%`（硬编码） | **`0.080%`** | 同上 |

- **worst case 0.253% 经复核正确，保留**：5.708 px / 该样本自身对角 2259.8 px（d0121，1880×1254）= 0.2526%。
  此前拟改的 0.371% 是「最大误差 ÷ 中位对角」，属分母混用，**已否决**。
- 脚本改动：`GLOBAL_REPL` 支持可调用值；新增 `_g_diag_pair()` / `_g_diag_i()` 实时计算，不再写死百分比。

## ② 插入 Figure 7 → §4.5.1

- 位置：Table 3 之后、讨论段（`The two arms are statistically indistinguishable…`）之前。
- 版式：居中图片 4.8 in × 5.40 in（正文可用宽 6.00 in），下接图题段落。
- 图题：`Figure 7. Ablation E1 at matched conditioning points: hard-paste baseline (left) versus marker injection with depth-conditioned ControlNet repainting (right). Each row reuses the same conditioning point in both arms; the ground-truth bounding box is overlaid in green.`
- 同步在 §4.5.1 引入段末尾追加交叉引用：` Figure 7 shows the two arms side by side at matched conditioning points.`
- 核验：内嵌媒体 6 → **7**（`word/media/image7.png` = 761,993 B，与磁盘 fig7 字节完全一致）。

## ③ 半像素越界率 → 98.0%

- 定义：`pos_err_px > 0.5`（包围盒整数像素量化 → 半像素界），实测 **294 / 300 = 98.0%**。
- v10 的 `71.3%` 无出处、不可复现（且与 `sub_pixel_rate=29.3%` 不能被同一分母解释）。
- 脚本改动：新增 `_g_beyond_half_px()`；`HALF_PX_BOUND = 0.5`。

## ④ 前导零格式统一 → `0.028 +/- 0.021`

- 全文带前导零 `0.x` 出现 **216** 次，无前导零 `.x` 仅 **2** 次，且这 2 次正是摘要的 `.028 / .021`。
- 已删除 `_a_abs_scale()` 中的 `.lstrip('0')`；v11 全文无前导零数字计数 = **0**。
- 现状：`0.028 +/- 0.021` 共 4 处（摘要 / §4.3 / §5 / Table 2），格式一致。

## ⑤ `figures/生成说明.md` 重写

- 旧版（2026-08-30）描述 20 对试产口径（7 图 + 3 表、位置误差 1.13 px、E1 IoU 0.699/0.653），与现行 300 对基准脱节。
- 已整体重写为 v11 口径：7 图 + **7 表**、两代执行器对照、当前权威数字表、重跑命令、过时产物清单。
- 旧版备份：`/tmp/生成说明.md.bak-20260830`。
- 顺带给 `figures/第四章自动报告.md` 加注 LEGACY 横幅（内容未改），防止被误引用。

## 附带修复

- 变更说明标题不再写死 `0-works-v9`，改为按输出文件名生成（`# 0-works-v11 变更说明`）。

## 校验结果（全部 PASS）

- 结构：段落 277 → **279**（+图 +图题）；表 7 → **7**（不变）；图片段落 6 → **7**
- 数值：`0.080%` × 3、`0.072%` × 0、`0.071%` × 0、`71.3%` × 0、`98.0%` × 1、`worst case 0.253%` × 1
- 残留：`pending` / `TODO` / `FIXME` / `[待` 均为 0
- 包完整性：`zip.testzip()` 无损坏；7 个 `a:blip` 的 rId → `word/media/*` 全部解析成功
- v10 → v11 段落级 diff 仅 5 处，均为上述预期改动

## 仍未处理（需你决策）

- `out_D` 下游代理 `R_any` 0.650 → 0.617 的正文解释（caveat 已有，未展开）
- `BG_MATCH=labels` 默认值与论文「随机采样 37,852」叙述的冲突
- `mountain` 场景约 300 张缺元数据；前端未暴露 `cnet_scales` / `use_lora`；三个 legacy 端点未冒烟

