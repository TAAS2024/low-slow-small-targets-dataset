# 3-生成端Agent搭建

> 🔑 **架构定位（2026-08-14 更新）**：论文叙事采用**双 Agent 对抗架构**，不再对外宣称「7-Agent」：
> - **生成端 Generator Agent**（本文档）：内部串联 7 个功能模块（modules），输入自然语言 → 输出 RGB-IR-BBox 三图一体；
> - **审查端 Validator Agent**（见 [[6-Validator设计与搭建]]）：执行 V1-V9 九阶段验证链，输出评分 + 失败码；
> - **对抗闭环**：Generator 产出 → Validator 审查 → 失败信号反馈 → Generator 修正（配课程硬化）。
>
> 本文档历史沿用的「Agent 1~7」称谓，现对应 Generator 内部的 7 个**功能模块（modules）**，不再作为独立 Agent 对外叙事。其中背景匹配（确定性检索）、IR 转换（确定性代码转换）等确定性环节不称 Agent。

> 更新日期：2026-08-10
> 版本：v7.2（2×5 网格 + bbox/IR 槽位；🎯 Focus 快速重定位；无人机红点拖拽标记）
> 状态：✅ Agent 1: LLM 语义解析 + Web 前端；✅ Agent 3: ControlNet v7 全管线（深度/分割/LoRA/CN重绘）+ Focus 重定位；✅ Agent 6: IR 转换集成；📐 Agent 2/4/7: 架构设计完成

---

## 一、生成端全景架构（v7.1）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│    低慢小数据集生成端 — 双 Agent 对抗架构（Generator 内部 7 modules）           │
│                                                                              │
│  用户: "阴天下午，四旋翼从远处飞近，仰拍"                                       │
│         │                                                                    │
│    ┌────▼─────┐                                                              │
│    │ Agent 1  │  LLM 语义解析（自然语言 → 双 JSON）                            │
│    │  (✅)     │  DeepSeek/OpenAI/Claude/Dry-Run + 缺失字段默认值              │
│    └────┬─────┘                                                              │
│         │  background_spec + full_scene_spec                                 │
│    ┌────▼─────┐                                                              │
│    │ Agent 2  │  背景匹配（JSON约束 → 37,852真实图库随机挑选）                     │
│    │  (✅)     │  scene_type 匹配 + 3 fallback                               │
│    └────┬─────┘                                                              │
│         │  真实背景图                                                         │
│    ┌────▼──────────────────────────────────────────────────────────┐         │
│    │ Agent 3: ControlNet v7 条件驱动全管线 (✅)                      │         │
│    │                                                                │         │
│    │  背景图 → MidasDepth → bg_depth                                │         │
│    │         → HeuristicSeg → bg_seg                                │         │
│    │                                                                │         │
│    │  每帧: LoRA txt2img 512² → drone_raw                           │         │
│    │       → rembg + dilate → drone_clean                           │         │
│    │       → silhouette 注入 bg_depth/bg_seg → drone_depth/seg      │         │
│    │       → ControlNet(depth+seg) img2img → final_rgb              │         │
│    │                                                                │         │
│    │  参数: skip_cnet=False(CN重绘)/True(仅alpha合成)                │         │
│    │  文件: condition_generator_v7.py + lora_inpainter_v7.py        │         │
│    │                                                                │         │
│    │  🔥 v7.2: Focus 快速重定位（🎯 按钮）                            │         │
│    │    跳过 LoRA 生成+rembg → 用已有 drone_rgba 重新合成+注入       │         │
│    │    API: POST /api/focus_drone/<sid>                            │         │
│    │    更新: drone_depth/seg → final_rgb → bbox → ir               │         │
│    └────┬───────────────────────────────────────────────────────────┘         │
│         │  10 产物: bg, bg_depth, bg_seg, drone_raw, drone_clean,            │
│         │          drone_depth, drone_seg, final_rgb, bbox, ir               │
│    ┌────▼─────┐                                                              │
│    │ Agent 4  │  Transformer 空间注入（📐 设计完成，暂由 Gaussian blob 替代）  │
│    │  (⏳)     │  背景depth/seg + 无人机 → 新depth/seg (含无人机)              │
│    └────┬─────┘                                                              │
│         │  （当前 v7.2 中此步骤在 Agent 3 内部以 Gaussian blob 方式执行）      │
│    ┌────▼─────┐                                                              │
│    │ Agent 6  │  IR 转换（RGB → 白热伪彩色 IR）✅ 已集成                      │
│    │  (✅)     │  rgb2ir_converter.py; 生成时 + Focus 时自动转换              │
│    └────┬─────┘                                                              │
│         │  RGB + IR 配对 + BBox                                              │
│    ┌────▼─────┐          ┌─────────────────────────────────────────┐         │
│    │ Agent 7  │─────────▶│  双层级 9 阶段验证                       │         │
│    │  调度器   │◀────────│  S1-5: 输入端预校验（22项规则）           │         │
│    │  (📐)     │         │  S6-S9: 输出端审查（质量/一致性/轨迹/检测）│         │
│    └────┬─────┘          └─────────────────────────────────────────┘         │
│         │               │                                                    │
│    ┌────▼─────┐          │                                                    │
│    │ 闭环反馈  │←─────────┘                                                    │
│    │  (📐)     │  失败码分派 → 对应环节参数调整 → 重新生成                      │
│    └──────────┘                                                              │
│                                                                              │
│  ✅ = 已完成    📐 = 架构设计完成，待实现    ⏳ = 部分实现                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 为什么是 Agent 架构？

生成端天然适合 Agent 化而非单一脚本：

| 特性 | 单一脚本 | Agent 架构 |
|:--|:--|:--|
| 独立可替换 | ❌ 改一处动全身 | ✅ 每个 Agent 独立接口 |
| 失败恢复 | ❌ 从头重跑 | ✅ 失败码定位，只重试出错环节 |
| 可观测性 | ❌ 黑盒 | ✅ 每步输出结构化日志 |
| 并行能力 | ❌ 串行 | ✅ 独立 Agent 可并行 |
| 人机协作 | ❌ | ✅ Human-in-the-loop 校准 |

---

## 二、Agent 1: LLM 语义解析 ✅ 🔥 v6.0 重写

### 2.1 职责

将用户自然语言描述解析为**双 JSON**，分别供背景匹配和全管线使用。

```
用户输入: "阴天下午，城市高楼背景，从下方仰拍"
          │
          ▼
    Agent 1: LLM
          │
    ┌─────┴─────┐
    ▼           ▼
 background   full_scene
 _spec.json   _spec.json
 (→ Agent 2)  (→ Agent 3/4/5/6)
```

### 2.2 双 JSON Schema

#### ❶ background_spec.json（Agent 2 消费，~200B）

| 字段 | 类型 | 默认 | 说明 |
|:--|:--|:--|:--|
| `scene_type` | enum | `puresky` | 12种：puresky/urban/rural/mountain/coastal/desert/forest/industrial/airfield/snow/night_city/residential |
| `time_of_day` | enum | `afternoon` | 6种：dawn/morning/noon/afternoon/dusk/night |
| `weather` | enum | `clear` | 8种：clear/cloudy/overcast/rainy/foggy/snowy/dusty/backlight |
| `camera_position` | enum | `bottom` | 5种：bottom/front/side/back/top |

#### ❷ full_scene_spec.json（Agent 3/4/5/6 + Validator 消费，~2KB）

```json
{
  "trajectory": [{"t":0, "action":"hover", "distance":75.0, "norm_u":0.5, "norm_v":0.3}],
  "background": {"scene_type":"puresky", "time_of_day":"afternoon", "weather":"clear", "scene_description":""},
  "camera": {"position":"bottom"},
  "meta": {"raw_input":"...", "confidence_note":"", "default_fills":["..."]}
}
```

### 2.3 v6.0 关键变更

| 移除 | 原因 |
|:--|:--|
| `drone_type` | 固定四旋翼（LoRA 仅有四旋翼数据） |
| `modality` | 始终 RGB→IR 转换，不在此处决策 |
| `camera.elevation_deg` | 背景是真实图片，透视由图片本身决定 |
| `camera.fov_deg` | 同上 |

| 新增 | 说明 |
|:--|:--|
| `scene_type` 新增 `puresky`/`snow`/`night_city`/`residential` | 覆盖更多场景，`puresky` 为默认 |
| `weather` 新增 `cloudy`/`snowy` | 细分云量，新增雪天 |
| `action` 拆分为 `circle_cw`/`circle_ccw` | 支持方向化盘旋 |
| `default_fills` 追踪 | 记录哪些字段使用了默认值，供闭环反馈参考 |
| 双 JSON 输出 | `extract_background_spec()` + `to_full_spec()` |

### 2.4 默认值表

| 字段 | 默认值 | 设计理由 |
|:--|:--|:--|
| `trajectory.action` | `hover` | 简单/稳定，无方向偏好 |
| `trajectory.distance` | `75m` | 中距，适配大多数 FOV |
| `trajectory.norm_u` | `0.5` | 画面水平居中 |
| `trajectory.norm_v` | `0.3` | 画面上方（天空区域），无人机不遮挡地面景物 |
| `scene_type` | `puresky` | 泛化性最强，无场景约束 |
| `time_of_day` | `afternoon` | 光照充足，对比度适中 |
| `weather` | `clear` | 无天气噪声，适合作为基线 |
| `camera.position` | `bottom` | 仰拍切合实际侦察场景（地面设备抬头拍摄低空无人机） |

### 2.5 地面方向硬约定

所有 camera position 下，**无人机底座始终朝向画面下方**（image v=1.0 方向）。不依赖 depth map 推断——因为纯天空背景（puresky）完全没有地面参照物。Agent 4 Transformer 渲染时直接硬编码。

### 2.6 实现：llm_parser.py v6.0

**路径**：`3-LLM starter/llm_parser.py`（695 行）
**代码结构**：

```
llm_parser.py
├── 0. 默认配置 DEFAULT_CONFIG (8项默认值)
├── 1. 枚举词典 (DroneAction/Weather/TimeOfDay/CameraPosition/SceneType)
│   └── 5组中英文关键词→枚举映射表 (~150条映射)
├── 2. 数据结构
│   ├── TrajectoryPoint: t/action/distance/norm_u/norm_v
│   ├── CameraSpec: v6.0仅保留 position（elevation/fov已移除）
│   └── SceneSpec: 全字段 + extract_background_spec() + to_full_spec()
├── 3. 解析器核心
│   ├── _parse_keywords(): 中文关键词优先匹配（无LLM开销）
│   └── _fill_defaults(): 缺失字段填充默认值 + 记录 default_fills
├── 4. LLM 后端适配
│   ├── _openai_call(): OpenAI/DeepSeek/Claude 统一调用
│   ├── _deepseek_call(): DeepSeek 专用
│   └── parse(): 关键词→dry-run→LLM 三层回退
└── 5. 顶层接口 parse_to_dual_json()

支持后端：DeepSeek（默认）/ OpenAI / Claude / Dry-Run
```

**API 接口**：
```python
from llm_parser import parse_to_dual_json

result = parse_to_dual_json("阴天下午城市高楼仰拍", backend="dry")
# result["background_spec"] → Agent 2 (4字段 ~200B)
# result["full_scene_spec"] → Agent 3/4/5/6 (~2KB)
# result["background_json"] → JSON string
# result["full_json"]        → JSON string
```

**CLI 使用**：
```bash
cd "3-LLM starter"
python3 llm_parser.py "阴天下午，城市高楼背景，从下方仰拍"          # 默认 DeepSeek
python3 llm_parser.py --backend dry "晴天黄昏，正面拍摄"           # Dry-run
python3 llm_parser.py --backend openai "foggy morning, drone hovering over forest"
```

**解析回退策略**（三层）：
1. 中文关键词直接匹配 `_parse_keywords()` → 无 LLM 开销
2. Dry-run 关键词匹配（如果指定 --dry）
3. LLM API 调用（DeepSeek/OpenAI/Claude）

### 2.7 背景图搜索模块：background_searcher.py

**路径**：`3-LLM starter/background_searcher.py`（149 行）
**职责**：从 37,852 张真实背景图库中根据 `background_spec` 搜索匹配背景图。

**图库**：`1-background-pool/curated_backgrounds/`
- 9 场景目录：airfield(3,854) / coastal(5,337) / desert(5,594) / forest(4,623) / industrial(3,117) / mountain(3,808) / pure_sky(2,696) / rural(5,092) / urban(3,751)
- 总计 ~38K 张图片

**场景类型映射与 fallback**：
| scene_type | 目录 | 说明 |
|:--|:--|:--|
| puresky | pure_sky | 纯天空 |
| urban / night_city | urban | night_city 无独立目录→fallback 到 urban |
| rural / residential | rural | residential 无独立目录→fallback 到 rural |
| mountain / snow | mountain | snow 无独立目录→fallback 到 mountain |
| coastal / desert / forest / industrial / airfield | 同名目录 | 一一对应 |

**核心函数**：
```python
search_background(background_spec, session_id) → {
    "ok": True,
    "image_url": "/images/session_0_bg.jpg",
    "source_path": "/abs/path/to/original.jpg",
    "scene_type": "urban",
    "dir_used": "urban"
}

get_placeholder_image(session_id, label, color) → {
    "ok": True,
    "image_url": "/images/session_0_placeholder_背景深度.png",
    "placeholder": True
}
```

`get_placeholder_image()` 使用 PIL 生成 400×400 纯色 + 居中文字标签的占位 PNG，用于尚未有真实结果的 slot。

---

## 三、Agent 2: 背景匹配 ✅ 基础版完成

### 3.1 职责

从 37,852 张真实背景图库中，根据 JSON 的 `background_spec` 字段匹配符合要求的背景图。

### 3.2 当前实现：background_searcher.py

**路径**：`3-LLM starter/background_searcher.py`（149 行）
**状态**：✅ 基础版完成 — 按 scene_type 匹配 + 随机选择

**已实现**：
- 9 场景目录 → 3 种 fallback（night_city→urban, residential→rural, snow→mountain）
- `search_background(spec, session_id)` → 随机选图 + 复制到 output_images
- `get_placeholder_image(sid, label, color)` → PIL 生成纯色占位图
- 与 web_app.py 无缝集成：解析后自动调用

**简化项**（v6.0 基础版 vs 设计）：
- 当前仅按 scene_type 过滤，未按 time_of_day/weather 精细化过滤
- 使用 `random.choice()` 而非设计中的 `random.seed(int(time.time()*14))`
- 无三层回退策略（精确→放宽→随机），目录不存在时直接 fallback

### 3.3 完整版待实现

```
JSON background_constraints: {scene_type, time_of_day, weather}
    ↓
读取 _global_index.jsonl → 三层过滤
    ↓
候选集 → random.seed(int(time.time()*14) % 2^31) → random.choice()
    ↓
回退策略: 精确匹配 → 放宽weather → 放宽tod → 放宽scene → 完全随机
```

**TODO**：
- [ ] 3.3.1 `_global_index.jsonl` 全局索引（合并9个场景的 CLIP 分类标签）
- [ ] 3.3.2 time_of_day / weather 维度过滤（需先完成 CLIP 分类标签提取）
- [ ] 3.3.3 确定性随机种子 `int(time.time()*14)`
- [ ] 3.3.4 三层回退策略

---

## 四、Agent 3: ControlNet v7 条件驱动全管线 ✅

### 4.1 职责

将 Agent 2 匹配的真实背景图，通过 LoRA 生成 + 标记注入 + ControlNet 重绘，产出完整的 8 槽中间产物。

### 4.2 管线（5 步）

```
背景图
  ├── MidasDetector(单例, 首次~15s) → bg_depth (深度图)
  ├── heuristic_segment() → bg_seg (分割图)
  │
  └── 每帧:
      1. LoRA txt2img 512² 白底 → drone_raw
      2. rembg + dilate → drone_clean (silhouette)
      3. drone silhouette 注入 bg_depth/bg_seg → drone_depth, drone_seg
      4. Alpha 合成 → init_image
      5. [skip_cnet?]
         ├─ False: ControlNet(depth+seg) img2img → 局部重绘 → final_rgb
         └─ True:  直接输出 alpha 合成 → final_rgb
```

### 4.3 当前实现

| 文件 | 行数 | 职责 |
|:--|:--|:--|
| `5-Controlnet/lora_inpainter_v7.py` | 696 | LoRA 生成 + rembg + ControlNet 重绘核心 |
| `3-LLM starter/condition_generator_v7.py` | 373 | 管线编排：背景→depth/seg→多帧→保存 |

**关键参数**：

| 参数 | 默认值 | 说明 |
|:--|:--|:--|
| `gen_seed` | `time_ns()/1000 + fi*777` | 微秒时钟 + 帧偏移 |
| `gen_steps` | 20 | LoRA txt2img 步数 |
| `blend_strength` | 0.65 | ControlNet img2img strength |
| `cnet_scales` | (0.85, 0.85) | depth/seg 条件强度 |
| `skip_cnet` | True (当前) | 跳过 CN 重绘 → 快速合成 |
| `use_lora` | True | LoRA v3 (Pocket_UAV trigger) |

**API 接口**：
```python
from condition_generator_v7 import generate_with_lora
result = generate_with_lora(bg_path, full_scene_spec, session_id, skip_cnet=True)
# → 8 产物: bg, bg_depth, bg_seg, drone_raw, drone_clean,
#           drone_depth, drone_seg, final_rgb
```

### 4.4 临时方案说明

- 深度提取：MidasDetector（controlnet_aux），单目深度。以后可升级为 DepthAnything V2
- 分割：启发式颜色阈值 + 空间位置规则。以后可升级为 SAM2/SegFormer
- Transformer 注入：当前以 Gaussian blob 在 Agent 3 内部实现，未来可能需要独立 Agent 4 接管

---

## 五、Agent 4: Transformer 空间注入 ⏳ 🔥 v7.1 延后

### 5.1 职责

将无人机空间信息注入到背景深度/分割图中，输出修改后的深度/分割图（含无人机 Gaussian blob）。

### 5.2 当前状态

v7.1 中此功能暂由 Agent 3 内部的 Gaussian blob 替代。`trajectory_to_frames()` 计算每帧无人机位置/尺寸/深度值，`_inject_markers()` 将 silhouette 注入 depth/seg。

未来独立 Agent 4 将用学习化的 CNN+Cross-Attention 替代手工 Gaussian blob。

### 5.3 输入/输出（目标）

**输入**：背景 depth map + 背景 seg map + 无人机位置(norm_u/v) + 距离 + 姿态 + 天气/时段
**输出**：修改后 depth map（背景深度 + 无人机深度值）+ 修改后 seg map（6超类 + 无人机=第7类）

### 5.4 架构（目标）

CNN双编码器(depth+seg) → 无人机Gaussian特征注入 → Cross-Attention融合 → 双头decoder(depth+seg)

---

## 六、Agent 5: LoRA 局部重绘 🔄 已融入 Agent 3

### 6.1 说明

v7.1 中，原本独立的 Agent 5（LoRA 局部 Inpainting）已作为 Agent 3 的内部步骤执行。详见 [[5-ControlNet场景生成管线]]。

LoRA 生成 + rembg 抠图 + ControlNet 重绘三阶段统一在 `lora_inpainter_v7.py` 中实现，不再需要独立 Agent。

### 6.2 为什么合并

三阶段（生成→抠图→重绘）天然紧密耦合，拆分反而增加编排复杂度。**Agent 3 现在就是「背景 → 最终 RGB」的完整黑盒**，输入背景图 + 场景 JSON，输出 8 个产物。

---

## 七、Agent 6: IR 域转换 📐

### 7.1 职责

Inpainting 后的 RGB 图 → `rgb2ir_converter.py` → 白热伪彩色 IR

与 v5.x 一致，保留不变。

---

## 八、Agent 7: Validator 调度器 + 闭环反馈 📐
     │
     ▼
 ControlNet 可消费的 Depth / Segmentation / Pose Map
```

### 3.4 与旧方案的区分

| | 旧方案（已废弃） | 新方案 |
|:--|:--|:--|
| 目标 | 预测 bbox 坐标 | 编码为条件向量 |
| 性质 | 生成器 | 编码器 |
| 问题 | 两次 mode collapse | 确定性映射，无 collapse 风险 |

### 3.5 ControlNet 条件图输出格式 (v4.9.2)

Agent 2 输出的 depth/seg map 中，无人机位置使用 **Gaussian 注意力区域**而非硬点标记：

| 特性 | 旧版 (v4.9) | 新版 (v4.9.2) |
|:--|:--|:--|
| seg 标记 | 4px 硬圆点 | Gaussian 软阈值 (weight>0.15) |
| depth 标记 | `depth×0.85` 微小扰动 | `bg×(1-w) + drone_depth×w` 物理混合 |
| σ 缩放 | 固定 | clamp(400/d, 3, 12)：近大远小 |
| 深度推导 | 无 | 1/(1+d/20)：30m→0.36, 100m→0.17 |
| ControlNet 可感知 | ❌ (0.02% 画面) | ✅ (185-1565px, 0.07-0.60%) |

详细数据见 [[4-Transformer时空编码#3.5 无人机 ControlNet 注意力区域设计]]。

### 3.6 30 帧连贯性验证 (demo_30frames.py)

```
测试用例: "阴天傍晚，四旋翼从远处飞近，仰拍观察工业厂房"
30 帧轨迹: approach(0-9) → hover(10-19) → lateral_move(20-29)
连贯性评分: 4/4 (depth平滑 + seg稳定 + 轨迹连续 + 动作多样)
输出: demo_30frames_output/ (grid_depth/segs + coherence_chart + report)
```

---

## 四、Agent 3: ControlNet 场景生成 📐

### 4.1 职责

接收 Agent 2 的条件向量，生成空间条件图（Depth/Seg/Pose），作为后续 LoRA 渲染的空间骨架。

### 4.2 核心分工

```
ControlNet → WHERE（空间骨架：目标在哪里、多大、什么姿态）
LoRA       → WHAT（外观填充：什么颜色、什么纹理、什么风格）
```

两者在同一管线中协同，非独立分支。

### 4.3 条件图类型

| 条件图 | 作用 | 来源 |
|:--|:--|:--|
| Depth Map | 目标深度/距离控制 | Agent 2 深度编码 |
| Segmentation Map | 目标区域掩码 | Agent 2 位置编码 |
| Pose Map | 目标姿态骨架 | Agent 2 姿态编码 |

---

## 五、Agent 4: 无人机 LoRA 渲染 📐

### 5.1 职责

在 ControlNet 空间条件图的引导下，使用无人机 LoRA 权重生成完整 RGB 场景。

### 5.2 与纯 LoRA 生成的对比

| | 纯 LoRA | ControlNet + LoRA |
|:--|:--|:--|
| 空间控制 | ❌ 不受控，目标位置随机 | ✅ 精确控制位置/尺度/姿态 |
| 背景一致性 | ❌ 每次生成不同背景 | ✅ 背景由条件图约束 |
| 批量生成 | ❌ 需人工筛选 | ✅ 自动化可重复 |

### 5.3 输入/输出

```
输入:  ControlNet 条件图 (depth + seg + pose) + 无人机 LoRA 权重
输出:  完整 RGB 场景（背景 + 无人机目标）
```

---

## 六、Agent 5: IR 域转换 📐

### 6.1 职责

将 Agent 4 生成的 RGB 图像转换为 IR 热红外伪彩色图像。

### 6.2 为什么是后处理而非生成？

SD 1.5 VAE 在自然 RGB 图像上训练，无法编码 IR 灰度图（三通道完全相同的像素落在 VAE 训练流形之外）。详见 [2-无人机LoRA训练.md](./2-无人机LoRA训练.md) 第六章。

### 6.3 转换方案

```
RGB 图像 → rgb2ir_converter.py → 白热 IR 伪彩色
                                 (温度强度映射 + 微蓝调)
```

---

## 七、Agent 6: 四阶段验证链 📐

### 7.1 职责

调度四阶段验证流水线，从粗到细逐层过滤不合格生成结果。

> **2026-08-04 更新**：S0-S3 的具体定义已重构为双模态适配版。完整设计见 [[7-持续学习循环设计#十三、审查端双模态架构与阶段任务规划]]。本节省略实现细节，只保留接口描述。

### 7.2 验证阶段（v2.0 双模态版）

```
┌─────────────────────────────────────────────────────────────┐
│                    四阶段验证链 (v2.0)                        │
│                                                             │
│  S0: 双模态图像质量                                          │
│  ├── RGB: BRISQUE + NIQE（主力）                             │
│  ├── IR:  像素范围 + 对比度零值 + FFT 伪影（低保真防御）      │
│  └── 失败 → 入 FailureBuffer                                │
│                                                             │
│  S1: 双模态一致性                                            │
│  ├── 单模态: RGB 尺寸 + RGB 光照 + IR bbox 对齐              │
│  ├── 跨模态: IoU(RGB_bbox, IR_bbox) > 0.95                 │
│  └── 失败 → 入 FailureBuffer                                │
│                                                             │
│  S2: 轨迹连续性                                              │
│  ├── 工具: 四个纯物理规则（位置/尺寸/加速度/方向）            │
│  ├── 永不可训，与模态无关                                     │
│  └── 失败 → 入 FailureBuffer                                │
│                                                             │
│  S3: 双模态检测有效性                                         │
│  ├── RGB: YOLO 推理 → IoU 判定（主力判 fail）                 │
│  ├── IR:  YOLO 推理 → 仅记录对照                             │
│  └── 失败 → 入 FailureBuffer                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、Agent 7: 闭环反馈 📐

### 8.1 职责

解析验证链输出的结构化失败码，分派到对应 Generator 组件进行精确微调。

> **2026-08-04 更新**：反馈路由从简单的"调参数重生成"升级为 **按失败阶段追溯到具体 Agent → 使用对应微调范式**。完整设计见 [[7-持续学习循环设计#十六、反馈闭环路由机制]]。

### 8.2 失败码路由表（v2.0）

| Validator 失败 | 责任组件 | 反馈目标 | 微调范式 |
|:--|:--|:--|:--|
| S0 RGB 质量差 | Agent 3（渲染问题） | Agent 3 LoRA | 范式 B：失败对比学习 |
| S1 尺寸不一致 | Agent 2（bbox 不合理） | Agent 2 Transformer | 范式 B：失败对比学习 |
| S1 光照不一致 | Agent 1（光照参数）+ Agent 3 | 需诊断 | 先人工判断再路由 |
| S2 轨迹物理异常 | Agent 2（轨迹生成） | Agent 2 Transformer | 范式 B：失败对比学习 |
| S3 RGB 检测不到 | Agent 3（渲染不够真实） | Agent 3 LoRA | 范式 B：失败对比学习 |
| S3 IR 检测不到 | YOLO 域适应不足 | 不反馈（对照线） | 不触发微调 |

### 8.3 闭环反馈流程

```
Validator 验证失败 → 记录失败码 + 失败帧
        │
        ▼
  FailureBuffer（按失败码聚类）
        │
        ▼
  堆积到阈值（S0:50条 / S2:20条 / S3:100条）
        │
        ▼
  路由到对应 Agent → 使用对应微调范式
        │
        ▼
  微调完成后 → A/B 评估 → 替换权重 / 回滚
```

**关键规则**：
- 不是每条失败都触发反馈——模式出现足够多次才触发
- S2 的 20 条阈值是最敏感的（物理规则违反是硬错误）
- S3 的 100 条阈值是最宽松的（YOLO 固有误检率）

### 8.4 与旧版的关键区别

| | 旧版 Agent 7 | 新版 Agent 7 |
|:--|:--|:--|
| 反馈方式 | 调整推理参数 → 重新生成 | 失败聚类 → 微调模型权重 |
| 模型能力 | 不变（冻结权重内搜索参数） | 持续进化（从错误中学习） |
| 粒度 | 逐条失败即响应 | 阈值触发 → 批量微调 |
| 路由 | 失败码 → 调参数 | 失败码 → 定位 Agent → 范武微调 |

---

## 九、Web 前端 ✅ v7.1

### 9.1 概述

Flask Web 服务，为生成管线提供可视化交互界面。当前已集成 Agent 1（LLM 解析 + 背景搜索）+ Agent 3（ControlNet v7 全管线），支持一键生成 + 单窗口重置级联。

### 9.2 技术栈

| 层 | 技术 | 文件 |
|:--|:--|:--|
| 后端 | Flask (Python) | `web_app.py` (495行) |
| 前端 | 原生 HTML/CSS/JS（无框架） | `templates/index.html` (1649行) |
| 管线编排 | condition_generator_v7.py | `condition_generator_v7.py` (536行) |
| LoRA+CN核心 | lora_inpainter_v7.py | `5-Controlnet/lora_inpainter_v7.py` (807行) |
| 背景搜索 | background_searcher.py | `background_searcher.py` (149行) |
| LLM | llm_parser.py | `llm_parser.py` (695行) |
| 运行地址 | `http://127.0.0.1:5000` | |

### 9.3 2×5 十图布局（v7.2 全新设计）

```
┌───────────────┬────────────────────────────────────────────────┐
│  会话记录      │          2×5 图片展示网格                        │
│  (左 280px)   │                                                │
│               │  ┌────────┬────────┬────────┬────────┬──────┐ │
│  [Demo] #1    │  │ 背景图  │背景深度 │背景Seg │LoRA生成│LoRA  │ │
│  🏙 urban     │  │  bg    │bg_depth│bg_seg  │drone_raw│抠图  │ │
│               │  ├────────┼────────┼────────┼────────┼──────┤ │
│  [Demo] #2    │  │无人机  │无人机  │最终    │BBox    │白热  │ │
│  🌲 forest    │  │深度图  │Seg图   │RGB     │标注    │红外  │ │
│               │  │drone_d │drone_s │final_rgb│bbox   │ir   │ │
│  ── 历史 ──   │  └────────┴────────┴────────┴────────┴──────┘ │
│  #3 真实会话  │                                                │
│               │  ↑ 放大 ◲◱  + 轨迹 ▸◂  + ↻ 重置             │
└───────────────┴────────────────────────────────────────────────┘
```

**十图 Slot 系统**：

| 行 | 列1 | 列2 | 列3 | 列4 | 列5 |
|:--|:--|:--|:--|:--|:--|
| 上 | `bg` 背景图 | `bg_depth` 背景深度 | `bg_seg` 背景分割 | `drone_raw` LoRA生成 | `drone_clean` LoRA抠图 |
| 下 | `drone_depth` 无人机深度 | `drone_seg` 无人机分割 | `final_rgb` 最终RGB | `bbox` BBox标注 | `ir` 白热红外 |

### 9.4 重置级联 ↻

每个窗口 hover 时显示橙色 ↻ 按钮，点击后该窗口及下游级联重新生成：

```
bg ──────────→ bg_depth, bg_seg ──→ drone_depth, drone_seg ──→ final_rgb
drone_raw ──→ drone_clean ────────→ drone_depth, drone_seg ──→ final_rgb
```

**实现**：
- 后端 `POST /api/regenerate/<sid>` — bg 重置时先随机换背景再全管线重跑
- 前端 `resetSlot(slot)` — 调 API → `renderImageGrid()` 刷新 → toast 提示联动窗口
- 图片 URL 带 `?t=<timestamp>` 强制浏览器刷新

### 9.5 Focus 快速重定位 🔥 v7.2 新增

**背景**：每次调无人机位置/大小后重新生成需要跑完整管线（LoRA 生成 + rembg 抠图），非常耗时。Focus 按钮跳过这两步，仅用已有的无人机 RGBA 重新合成 + 注入 + 保存下游 slot。

**实现**：

| 文件 | 改动 |
|:--|:--|
| `5-Controlnet/lora_inpainter_v7.py` | `inpaint()` 返回值新增 `drone_rgba`；新增 `reposition()` 方法（跳过 Step1/2，直接用已有 drone_rgba 做合成+注入+重绘） |
| `3-LLM starter/condition_generator_v7.py` | 保存 `drone_rgba` 到磁盘；新增 `focus_reposition()` 函数 |
| `3-LLM starter/web_app.py` | 新增 `POST /api/focus_drone/<sid>` 端点 |
| `3-LLM starter/templates/index.html` | 滑块旁新增 🎯 Focus 按钮 + `focusDrone()` JS |

**坐标转换**：

| 前端值 | 后端值 | 公式 |
|:--|:--|:--|
| `dronePos.x` (0-100%) | `nu` | `x / 100` |
| `dronePos.y` (0-100%) | `nv` | `y / 100` |
| `droneScale` (0.03-0.50) | `target_px` | `clip(scale × 192, 12, 96)` |
| `droneScale` (0.03-0.50) | `dval` | `clip(scale × 1.4, 0.15, 0.7)` |

**管线差异**：

| 步骤 | 首次生成 | Focus 重定位 |
|:--|:--|:--|
| MidasDepth → bg_depth | ✅ 执行 | ❌ 跳过（复用） |
| HeuristicSeg → bg_seg | ✅ 执行 | ❌ 跳过（复用） |
| LoRA txt2img → drone_raw | ✅ 执行 | ❌ 跳过（复用 drone_rgba） |
| rembg + dilate → drone_clean | ✅ 执行 | ❌ 跳过（复用 drone_rgba） |
| Silhouette 注入 depth/seg | ✅ 执行 | ✅ 执行（新位置） |
| Alpha 合成 | ✅ 执行 | ✅ 执行（新位置） |
| ControlNet 重绘 | 可选 | 可选 |
| IR 转换 | ✅ 执行 | ✅ 执行 |
| BBox 提取 | ✅ 执行 | ✅ 执行 |

**API 端点**：

```
POST /api/focus_drone/<sid>
Body: {"dronePos": {"x": 50, "y": 30}, "droneScale": 0.15}
Response: {"ok": true, "images": {...}, "nu": 0.5, "nv": 0.3}
```

**使用流程**：
1. 生成场景 → 背景图左上出现红点标记 + 滑块面板
2. 拖拽红点调整位置，拖动滑块调整大小
3. 点击 🎯 Focus → 后端 `focus_reposition()` 重定位
4. 下游 slot 自动更新：`drone_depth` / `drone_seg` / `final_rgb` / `bbox` / `ir`
5. 不重新生成：LoRA 无人机、rembg 抠图、背景深度/分割分析

### 9.6 Seed 机制

```python
gen_seed = int(time.time_ns() / 1000 + fi * 777) % (2**31)
blend_seed = gen_seed + 100_000
```

微秒级时钟 + 帧偏移，确保跨请求和跨帧 seed 永不重复。详见 [[5-ControlNet场景生成管线#五、Seed 机制]]。

### 9.7 API 接口

| 端点 | 方法 | 说明 |
|:--|:--|:--|
| `/` | GET | Web 前端页面 |
| `/api/parse` | POST | 解析自然语言 → 双 JSON + 自动搜索背景 + 创建会话 + 一键生成 |
| `/api/search-background/<sid>` | POST | 手动重新搜索背景图 |
| `/api/regenerate/<sid>` | POST | 🔥 重置指定 slot 及下游 |
| `/api/focus_drone/<sid>` | POST | 🎯 快速重定位无人机（跳过 LoRA 生成） |
| `/api/sessions` | GET | 获取全部会话列表 |
| `/api/session/<id>` | GET | 获取单条会话详情 |
| `/api/session/<id>/images` | POST | 上传生成图（多文件，每 slot 可传多帧） |
| `/api/shutdown` | POST | 关闭 Flask 服务 |
| `/images/<filename>` | GET | 静态图片服务 |

### 9.8 文件结构

```
3-LLM starter/
├── llm_parser.py              # LLM 解析核心 (695行)
├── web_app.py                 # Flask 服务 (495行)
├── background_searcher.py     # 背景图搜索模块 (149行)
├── condition_generator_v7.py  # 🔥 管线编排 (536行) — v7.2
├── condition_generator.py     # 旧版 (280行) — v6，保留兼容
├── .env / .env.example        # API Key 配置
├── templates/
│   └── index.html             # 前端页面 (1649行)
├── sessions/                  # 会话持久化
└── output_images/             # 生成图片存储
```

---

## 十、关键设计决策

### 10.1 架构演进

```
v2.0: 双分支 ControlNet（分割 + 深度各自独立）
  └→ 架构过于复杂，两条分支协同困难

v3.0: 多 LoRA 融合（无 ControlNet）
  └→ 空间不可控，生成位置随机

v4.0: LLM → Transformer → ControlNet + LoRA
  └→ ✅ 自然语言驱动 + 空间可控 + 外观解耦

v4.3: 简化为单 LoRA + IR 后处理
  └→ IR 背景 LoRA 废弃，改为 rgb2ir 伪彩色

v4.4: Web 前端上线
  └→ Agent 1 可用化，为全链路提供可视化入口

v6.0: ControlNet 移除 → 真实背景 + 深度/分割提取 + LoRA 局部重绘
  └→ 背景从合成改为 37,852 张真实图库；6 格前端 + 条件图生成

v7.0: ControlNet 回归 — depth+seg 双条件驱动局部重绘
  └→ 标记注入 → CN 重绘；8 格前端 + LoRA v3

v7.1: 重置级联 + 时钟 seed + skip_cnet 模式 + 前端 cache-busting
  └→ 当前版本

v7.2: 2×5 网格 + bbox/IR 槽位 + 🎯 Focus 快速重定位 + 无人机红点拖拽标记
  └→ 🔥 当前版本
```

### 10.2 设计原则

| 原则 | 说明 |
|:--|:--|
| 模块化 | 每个 Agent 独立接口，可单独开发/测试/替换 |
| 先跑通主线 | 单机型、单场景优先，验证全链路可行性后再扩展 |
| 按需生成 | 不做预计算仓库，选中背景后当场计算 depth/seg |
| 可观测性 | 8 槽中间产物可见，每步输出结构化日志 |

## 十一、当前状态与后续计划

### 11.1 完成项

| 组件 | 状态 | 产出 |
|:--|:--|:--|
| Agent 1: LLM 解析 | ✅ | `llm_parser.py` + 双 JSON + 4 后端 |
| Agent 2: 背景匹配 | ✅ | `background_searcher.py` + 9 场景 + 3 fallback |
| Agent 3: ControlNet v7 管线 | ✅ | `condition_generator_v7.py` + `lora_inpainter_v7.py` |
| Web 前端 | ✅ | Flask @ 127.0.0.1:5000，10 格 + 重置级联 + Focus 重定位 |
| 无人机 LoRA | ✅ | v3: Pocket_UAV trigger, 2000 step |
| 重置级联 | ✅ | `POST /api/regenerate` + ↻ 按钮 |
| Focus 快速重定位 | ✅ | `POST /api/focus_drone` + 🎯 按钮 (v7.2) |
| 时钟 Seed | ✅ | `time_ns()/1000 + fi*777` + 前端 cache-busting |
| IR 转换集成 | ✅ | `rgb2ir_converter.py` — 生成+Focus时自动转换 |
| BBox 提取 | ✅ | segmentation class=7 提取 + OpenCV 可视化 |

### 11.2 待实现

| 组件 | 优先级 | 依赖 |
|:--|:--|:--|
| Agent 4: Transformer 注入 | 中 | Agent 3 ✅ |
| Agent 7: Validator 集成 | 中 | Agent 3 ✅ |
| ControlNet 重绘调参 | 中 | skip_cnet=False 模式调试 |
| 背景匹配精细化过滤 | 低 | Agent 2 ✅ |
| 闭环反馈 | 低 | Agent 6 + 7 |

## 十二、版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v3.1 | 2026-08-10 | 🔥 v7.2：2×5 网格(bbox/IR) + 🎯 Focus 快速重定位 + 无人机红点拖拽标记；Agent 6 IR 集成；Web 前端 1649行 |
| v3.0 | 2026-08-09 | 🔥 v7.1 架构：ControlNet depth+seg 双条件回归；8 格前端 + 重置级联 + 时钟 seed；Agent 5 合并入 Agent 3 |
| v2.3 | 2026-08-07 | Agent 3 基础版：condition_generator.py；Web 前端 6 格 |
| v2.2 | 2026-08-07 | 全局代码清理：Controlnet/Transformer/Lora 旧代码归档 |
| v2.1 | 2026-08-07 | Agent 2：background_searcher.py；Web 前端 v6.0 重构 |
| v2.0 | 2026-08-07 | v6.0 架构：LLM 双 JSON；背景搜图；6 格前端 |
| v1.3 | 2026-08-04 | Agent 6/7 双模态验证 + 反馈路由 |
| v1.0 | 2026-07-30 | 初版：7-Agent 架构 + Agent 1 + Web 前端 |
