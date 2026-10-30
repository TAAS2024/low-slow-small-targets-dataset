# 5-ControlNet(depth+seg) 条件驱动局部重绘管线

> 更新日期：2026-08-10
> 版本：v7.2 — 2×5 网格 + bbox/IR + 🎯 Focus 快速重定位
> 状态：✅ 全管线打通，Web 前端集成，10 槽观察 + 重置级联 + Focus 重定位

---

## 一、v7 管线全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ControlNet v7 — LoRA + 标记注入 + CN 重绘                    │
│                                                                         │
│  背景图 ──→ MidasDepth ──→ bg_depth (深度图)                             │
│     │          │                                                        │
│     │    HeuristicSeg ──→ bg_seg (分割图)                                │
│     │                                                                   │
│     └──→ 每个无人机帧:                                                    │
│              │                                                          │
│       ┌──────┴──────┐                                                   │
│       ▼             ▼                                                   │
│   Step 1: LoRA txt2img 512² 白底 → drone_raw                            │
│       │                                                                 │
│       ▼                                                                 │
│   Step 2: rembg + dilate → drone_clean (silhouette)                     │
│       │                                                                 │
│       ▼                                                                 │
│   Step 3: drone silhouette 注入 bg_depth/bg_seg → drone_depth, drone_seg│
│       │                                                                 │
│       ├──→ Alpha 合成 → init_image (纹理起点)                            │
│       │                                                                 │
│       └──→ ControlNet(depth=drone_depth, seg=drone_seg) img2img          │
│            → 只在 drone 区域应用结果 → final_rgb                          │
│                                                                         │
│  产物: bg, bg_depth, bg_seg, drone_raw, drone_clean,                     │
│         drone_depth, drone_seg, final_rgb                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心理念

**先标记，再重绘**。不同于简单贴入，v7 将无人机轮廓注入到背景的 depth/seg 图中，让 ControlNet 把无人机区域当作「场景的一部分」来重绘。这样生成的无人机光照和纹理与背景自然融合。

---

## 二、实现文件

| 文件 | 行数 | 职责 |
|:--|:--|:--|
| `5-Controlnet/lora_inpainter_v7.py` | 807 | LoRA 生成 + rembg + ControlNet 重绘核心 + reposition() |
| `3-LLM starter/condition_generator_v7.py` | 536 | 管线编排：背景加载 → depth/seg → 多帧调用 inpainter + focus_reposition() |

### 2.1 lora_inpainter_v7.py 核心函数

| 函数 | 作用 |
|:--|:--|
| `load()` | 加载 SD1.5 + ControlNet(depth+seg) + LoRA v3 权重 |
| `_generate(seed, ...)` | LoRA txt2img 生成 drone_raw (512² 白底) |
| `_extract_drone(img)` | rembg 抠图 → dilate → drone silhouette |
| `_inject_markers(...)` | drone silhouette 注入 bg_depth/bg_seg → 新条件图 |
| `_blend(...)` | ControlNet img2img 局部重绘 → 羽化贴回 |
| `inpaint(...)` | 完整管线入口，参数含 skip_cnet 开关 |

### 2.2 关键参数

| 参数 | 默认值 | 说明 |
|:--|:--|:--|
| `gen_seed` | 时钟派生 | 见 [[#五、Seed 机制\|seed 机制]] |
| `gen_steps` | 20 | LoRA txt2img 采样步数 |
| `blend_seed` | gen_seed + 100000 | ControlNet 重绘 seed |
| `blend_steps` | 15 | img2img 重绘步数 |
| `blend_strength` | 0.65 | img2img denoising strength |
| `guidance` | 7.5 | CFG scale |
| `cnet_scales` | (0.85, 0.85) | depth/seg ControlNet 条件强度 |
| `skip_cnet` | False | 跳过 ControlNet → 仅 alpha 合成输出 |

---

## 三、10 槽观察窗口

Web 前端展示完整管线的 10 个产物（2×5 网格）：

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  背景图   │ 背景深度  │ 背景Seg   │ LoRA生成  │ LoRA抠图  │
│   bg     │ bg_depth │  bg_seg  │ drone_raw│drone_clean│
├──────────┼──────────┼──────────┼──────────┼──────────┤
│无人机深度 │ 无人机Seg │ 最终RGB   │ BBox标注  │ 白热红外  │
│drone_depth│drone_seg │final_rgb │   bbox   │    ir    │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

| Slot | 内容 | 多帧 | 说明 |
|:--|:--|:--|:--|
| `bg` | 背景图 | ❌ | 从 curated_backgrounds 搜索 |
| `bg_depth` | 背景深度图 | ❌ | MidasDetector 提取 |
| `bg_seg` | 背景分割图 | ❌ | 启发式分割 |
| `drone_raw` | LoRA 原始生成 | ✅ | 512² 白底 LoRA 生成 |
| `drone_clean` | LoRA 抠图 | ✅ | rembg + dilate |
| `drone_depth` | 无人机标记深度 | ✅ | silhouette 注入深度 |
| `drone_seg` | 无人机标记分割 | ✅ | silhouette 注入分割 |
| `final_rgb` | 最终合成 RGB | ✅ | ControlNet 重绘或 alpha 合成 |
| `bbox` | BBox 标注可视化 | ✅ | seg class=7 提取 + cv2.rectangle (v7.2) |
| `ir` | 白热红外图 | ✅ | rgb2ir 伪彩色转换 (v7.2) |

---

## 四、重置级联

每个窗口有 ↻ 按钮（hover 显示），点击后该窗口及下游级联重新生成。

```
bg ──────────→ bg_depth, bg_seg ──→ drone_depth, drone_seg ──→ final_rgb
drone_raw ──→ drone_clean ────────→ drone_depth, drone_seg ──→ final_rgb
```

**实现**：`web_app.py` `POST /api/regenerate/<sid>` + 前端 `resetSlot(slot)`。
- bg 重置：先 `search_background()` 随机换背景 → 全管线重跑
- 其他 slot：重用当前背景 → 全管线重跑（新 seed）
- 返回 `regenerated` 列表标明实际更新的窗口

## 四-B、Focus 快速重定位 🔥 v7.2 新增

跳过 LoRA 生成 + rembg，仅用已有 drone_rgba 在新的位置/大小重新合成 + 注入 + 保存下游。

### 实现文件

| 文件 | 改动 |
|:--|:--|
| `lora_inpainter_v7.py` | `inpaint()` 返回 `drone_rgba`；新增 `reposition()`（807行） |
| `condition_generator_v7.py` | 保存 `drone_rgba`；新增 `focus_reposition()`（536行） |
| `web_app.py` | 新增 `POST /api/focus_drone/<sid>`（495行） |
| `index.html` | 🎯 按钮 + `focusDrone()` JS（1649行） |

### reposition() 方法签名

```python
def reposition(self, background_bgr, bg_depth, bg_seg, drone_rgba,
               nu=0.5, nv=0.5, target_px=70, dval=0.4, class_id=7,
               blend_seed=123, blend_steps=15, blend_strength=0.65,
               guidance=7.5, crop_pad_ratio=2.5, feather_px=12,
               cnet_scales=(0.85, 0.85),
               time_of_day="afternoon", weather="clear",
               use_lora=True, skip_cnet=False)
```

内部流程：确认 drone_rgba → 提取 bbox → `_composite()` → `inject_into_conditions()` → `_controlnet_repaint()`（或跳过）

### 坐标转换

| 前端值 | 后端值 | 公式 |
|:--|:--|:--|
| `dronePos.x` (0-100%) | `nu` | `x / 100` |
| `dronePos.y` (0-100%) | `nv` | `y / 100` |
| `droneScale` (0.03-0.50) | `target_px` | `clip(scale × 192, 12, 96)` |
| `droneScale` (0.03-0.50) | `dval` | `clip(scale × 1.4, 0.15, 0.7)` |

### 跳过的步骤

| 步骤 | Focus 重定位 |
|:--|:--|
| MidasDepth → bg_depth | ❌ 复用 |
| HeuristicSeg → bg_seg | ❌ 复用 |
| LoRA txt2img → drone_raw | ❌ 复用 drone_rgba |
| rembg + dilate → drone_clean | ❌ 复用 drone_rgba |
| Silhouette 注入 depth/seg | ✅ 执行（新位置） |
| Alpha 合成 | ✅ 执行（新位置） |
| ControlNet 重绘 | 可选 |
| IR 转换 | ✅ 执行 |
| BBox 提取 | ✅ 执行 |

---

## 五、Seed 机制

所有随机性由时钟驱动，确保每次生成结果不同：

```python
gen_seed = int(time.time_ns() / 1000 + fi * 777) % (2 ** 31)
blend_seed = gen_seed + 100_000
```

- `time_ns()/1000`：微秒级时钟，跨请求永不重复（1ms 间隔 → seed 差 ~1000）
- `fi * 777`：帧偏移，同一请求内相邻帧 seed 差 777，保证每帧独立
- `blend_seed` 加 10 万避免与 gen_seed 碰撞

前端图片渲染加 `?t=Date.now()` 强制刷新，防止浏览器缓存旧图。

---

## 六、skip_cnet 模式

当 `skip_cnet=True` 时，跳过 ControlNet 重绘步骤，仅将 LoRA 抠图 alpha 合成到背景。用于快速迭代和调试。

当前 Web 前端默认使用 `skip_cnet=True`（生成速度快 ~5 倍），ControlNet 重绘可手动开启。

---

## 七、6 超类颜色映射（沿用 v5）

| 索引 | 超类 | RGB 颜色 | 色块 |
|:--|:--|:--|:--|
| 0 | sky | (128,192,255) | 浅蓝 |
| 1 | tree | (0,128,0) | 绿色 |
| 2 | building | (128,128,128) | 灰色 |
| 3 | mountain | (139,90,43) | 棕色 |
| 4 | water | (30,144,255) | 蓝色 |
| 5 | ground | (200,180,140) | 土色 |
| 7 | drone | (255,0,0) | 红色（v7 新增） |

---

## 八、历史记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v7.2 | 2026-08-10 | 🔥 10 槽网格(bbox/IR) + 🎯 Focus 快速重定位(reposition) + 无人机红点拖拽标记 |
| v7.1 | 2026-08-09 | 🔥 完整重写：LoRA v3 + ControlNet depth/seg 重绘管线落地；8 槽前端 + 重置级联 + 时钟 seed |
| v6.0 | 2026-08-07 | 误标"已废弃"——ControlNet 在 v6.0 中实际以 Inpainting 形式保留 |
| v5.x | 2026-08-03 | Dual ControlNet 全图生成（已归档至 `5-Controlnet/archive/`） |

---

## 九、代码归档

v5.x 旧代码位于 `5-Controlnet/archive/`：
- `dual_controlnet_pipeline.py` / `full_pipeline.py`
- `generate_scenes.py` / `compare_ablation.py`
- `drone_compositor.py` / `transformer2controlnet.py`
- `test_frames_30/`

v7 当前代码：
- `5-Controlnet/lora_inpainter_v7.py` — 核心
- `3-LLM starter/condition_generator_v7.py` — 编排
- `3-LLM starter/web_app.py` — 前端 API
- `3-LLM starter/templates/index.html` — 前端界面
