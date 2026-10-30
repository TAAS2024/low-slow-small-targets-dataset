# dataset_lss_v1 — 低慢小 RGB-IR 数据集（v9）

- 组数：200；每组 10 帧；图像 2000 张
- 模态：RGB + 伪红外（IR），共享同一 bbox 标注
- 生成路径：第 1 帧完整管线；第 2-10 帧走 Focus（FRRM）缓存复用路径
- 计时：基线帧 17.38 s，Focus 帧 9.23 s
  （平均加速 1.89×）
- 总耗时：18779.4 s

## 目录

```
groups/gNNNN/
  bg.png bg_depth.png bg_seg.png      背景与条件图（组级，仅算一次）
  meta.json                           组元数据 + 逐帧记录
  frames/fNN_rgb.png                  最终 RGB
          fNN_ir.png                  伪红外
          fNN_bbox.png                标注可视化
          fNN_rgba.png                该帧无人机 RGBA（前抠图）
          fNN_depth.png fNN_seg.png    注入的 GT 深度 / 分割
          fNN_clean.png               干净无人机图层
annotations.json  manifest.json  groups.jsonl  summary.json
```

## 游走规律

`sweep` 水平之字 / `approach` 单调接近 / `retreat` 单调远离 / `orbit` 绕心圆周 /
`cross` 对角穿越 / `dwell` 悬停微抖；位置限制在天空区 u∈[0.12,0.88]、v∈[0.10,0.68]。
每组规律由 seed 决定，组内目标身份由固定 `identity_seed` 锁定。
