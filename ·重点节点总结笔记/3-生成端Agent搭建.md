# 3-生成端Agent搭建

> 更新日期：2026-07-30
> 版本：v1.1（Step 1 完成 + 全功能测试通过 + JSON 持久化）
> 状态：✅ Step 1: LLM 语义解析 + Web 前端 + 测试；📐 Step 2-7: 架构设计完成

---

## 一、生成端全景架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    低慢小数据集生成端 — 7-Agent 协同架构                        │
│                                                                              │
│  用户: "阴天下午，四旋翼从远处飞近，仰拍"                                       │
│         │                                                                    │
│    ┌────▼─────┐                                                              │
│    │ Agent 1  │  LLM 语义解析（自然语言 → 结构化 JSON）                        │
│    │  (✅)     │  DeepSeek/OpenAI/Claude/Dry-Run                              │
│    └────┬─────┘                                                              │
│         │  9 字段 Schema                                                     │
│    ┌────▼─────┐                                                              │
│    │ Agent 2  │  Transformer 时空编码（JSON → 逐帧 ControlNet 条件向量）       │
│    │  (📐)     │  位置/深度/姿态/天气时段/相机参数 五模块编码                    │
│    └────┬─────┘                                                              │
│         │  条件向量                                                           │
│    ┌────▼─────┐                                                              │
│    │ Agent 3  │  ControlNet 场景生成（条件向量 → Depth/Seg/Pose map）          │
│    │  (📐)     │  SD1.5 + 多条件 ControlNet                                   │
│    └────┬─────┘                                                              │
│         │  空间骨架                                                           │
│    ┌────▼─────┐                                                              │
│    │ Agent 4  │  无人机 LoRA 渲染（空间骨架 + 无人机外观 → 完整 RGB）          │
│    │  (📐)     │  ControlNet 决定 WHERE，LoRA 决定 WHAT                        │
│    └────┬─────┘                                                              │
│         │  RGB 图像                                                          │
│    ┌────▼─────┐                                                              │
│    │ Agent 5  │  IR 转换（RGB → 白热伪彩色 IR）                                │
│    │  (📐)     │  rgb2ir_converter.py                                         │
│    └────┬─────┘                                                              │
│         │  RGB + IR 配对                                                     │
│    ┌────▼─────┐          ┌─────────────────────────────────────────┐         │
│    │ Agent 6  │─────────▶│  四阶段验证链                            │         │
│    │  调度器   │◀────────│  S0: 背景物理校验（二分类器）              │         │
│    │  (📐)     │         │  S1: CLIP 语义一致性（视角对齐）          │         │
│    └────┬─────┘         │  S2: YOLO 目标检测（可检测性校验）        │         │
│         │               │  S3: IQA 无参考质量（模糊/伪影）          │         │
│    ┌────▼─────┐          └─────────────────────────────────────────┘         │
│    │ Agent 7  │  闭环反馈 —— 失败码 → 参数调整 → 重新生成                      │
│    │  (📐)     │  S0_BG_UNREALISTIC → 重生成背景                               │
│    └──────────┘  S2_POSITION_OFFSET → 调整位置编码                             │
│                                                                              │
│  ✅ = 已完成    📐 = 架构设计完成，待实现                                       │
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

## 二、Agent 1: LLM 语义解析 ✅

### 2.1 职责

将用户自然语言描述解析为结构化 9 字段 JSON，作为后续全管线的唯一输入。

### 2.2 9 字段 Schema

| 字段 | 类型 | 说明 | 示例 |
|:--|:--|:--|:--|
| `drone_type` | enum | 无人机类型 | `quadrotor` |
| `trajectory[]` | array | 逐帧序列 `{t, action, distance, norm_u, norm_v}` | `[{t:0, action:"hover", distance:100, ...}]` |
| `time_of_day` | enum | 时段 | `dawn / morning / afternoon / dusk / night` |
| `weather` | enum | 天气 | `clear / overcast / rainy / foggy / dusty / backlight` |
| `scene_type` | enum | 场景类型 | `urban / rural / mountain / coastal / desert / forest / industrial / airfield` |
| `scene_description` | string | ControlNet prompt 风格英文描述 | `"An overcast afternoon sky above a city skyline..."` |
| `modality` | enum | 输出模态 | `RGB` / `IR` |
| `camera` | object | 相机参数 | `{position, elevation_deg, fov_deg}` |
| `confidence_note` | string | 置信度备注 | `[dry-run] 未调用 LLM` |

### 2.3 相机位置语义（关键设计）

以**无人机为参照中心**，相机位置描述从无人机看出去的视角：

```
         top (俯拍，看见背部)
          ↑
          │
  back ←─ 🚁 ─→ front (正面拍摄)
          │
          ↓
        bottom (仰拍，看见腹部)
```

| position | 相机位置 | 拍摄角度 | 可见部位 |
|:--|:--|:--|:--|
| `bottom` | 相机在无人机下方 | 仰拍 | 腹部、桨叶底面 |
| `top` | 相机在无人机上方 | 俯拍 | 背部、桨叶顶面 |
| `front` | 相机在无人机前方 | 正面 | 机头、前臂 |
| `side` | 相机在无人机侧方 | 侧面 | 侧臂、机身 |
| `back` | 相机在无人机后方 | 尾部 | 尾灯、后臂 |

> 此设计与 Tiger 的数据打标逻辑一致：bottom=仰拍切合实际场景（地面设备抬头拍摄低空无人机）。

### 2.4 实现：llm_parser.py

**路径**：`3-LLM starter/llm_parser.py`

**支持后端**：

| 后端 | 标识 | 说明 |
|:--|:--|:--|
| DeepSeek | `deepseek` | 默认，兼容 OpenAI SDK（base_url=https://api.deepseek.com） |
| OpenAI | `openai` | GPT-4o |
| Claude | `claude` | Anthropic |
| Dry-Run | `dry` | 离线测试，关键词映射 + 默认值填充 |

**核心设计**：
- 枚举词典 + 中文关键词映射（如「四旋翼」→ `quadrotor`，「仰拍」→ `bottom`）
- Few-shot prompt 含 5 个示例覆盖常见场景组合
- 自动处理 markdown 代码块包裹、枚举大小写匹配
- CLI 支持：`python3 llm_parser.py "阴天下午城市高楼四旋翼飞近仰拍"`

**CLI 使用**：
```bash
cd "3-LLM starter"

# DeepSeek（默认）
python3 llm_parser.py "晴天上午，四旋翼在城市上空悬停，正面拍摄"

# Dry-run 测试
python3 llm_parser.py --dry "阴天下午城市高楼四旋翼飞近仰拍"

# 切换后端
python3 llm_parser.py --backend openai "foggy morning, drone hovering over forest"
```

**输出示例**：
```json
{
  "drone_type": "quadrotor",
  "trajectory": [
    {"t": 0.0, "action": "approach", "distance": 100.0,
     "norm_u": 0.5, "norm_v": 0.8}
  ],
  "time_of_day": "afternoon",
  "weather": "overcast",
  "scene_type": "urban",
  "scene_description": "Overcast afternoon above a modern city skyline...",
  "modality": "RGB",
  "camera": {
    "position": "bottom",
    "elevation_deg": 30.0,
    "fov_deg": 60.0
  },
  "confidence_note": ""
}
```

---

## 三、Agent 2: Transformer 时空编码 📐

### 3.1 职责

将 Agent 1 的结构化 JSON 展开为逐帧 ControlNet 条件向量，实现「自然语言 → 空间控制」的桥梁。

### 3.2 五个编码模块

```
JSON (Agent 1 输出)
     │
     ├──→ ① 位置编码 (Position Encoder)
     │       trajectory[].norm_u/v → 目标在画面中的归一化坐标
     │       输出: position_embedding ∈ R^d
     │
     ├──→ ② 深度编码 (Depth Encoder)
     │       trajectory[].distance → 目标像素占比
     │       50m 远处占 5%，200m 近处占 30%
     │       输出: scale_factor ∈ R
     │
     ├──→ ③ 姿态编码 (Pose Encoder)
     │       trajectory[].action → 飞行姿态角 → 关键点偏移向量
     │       输出: pose_offset ∈ R^(k×2)
     │
     ├──→ ④ 天气/时段编码 (Weather-Time Encoder)
     │       weather + time_of_day → learnable embedding
     │       输出: env_embedding ∈ R^d
     │
     └──→ ⑤ 相机参数编码 (Camera Encoder)
             camera.fov + camera.elevation → 投影矩阵
             输出: proj_matrix ∈ R^(3×3)
```

### 3.3 融合与输出

```
小型 MLP/CNN 融合
     │
     ▼
 逐帧条件向量 (per-frame conditioning vector)
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

### 7.2 验证阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    四阶段验证链                               │
│                                                             │
│  Stage 0: 背景物理校验                                       │
│  ├── 工具: ResNet/EfficientNet 二分类器                      │
│  ├── 训练: 576 帧真实 IR 背景 vs 生成 IR 背景                 │
│  └── 失败码: S0_BG_UNREALISTIC → 重生成背景                   │
│                                                             │
│  Stage 1: CLIP 语义一致性                                    │
│  ├── 工具: CLIP ViT-L/14                                    │
│  ├── 检查: 生成图与 Agent 1 场景描述的语义对齐                │
│  └── 失败码: S1_SEMANTIC_MISMATCH → 调整 LLM prompt           │
│                                                             │
│  Stage 2: YOLO 目标检测                                      │
│  ├── 工具: YOLOv8 (Anti-UAV410 fine-tune)                    │
│  ├── 检查: 无人机目标可检测性 (confidence > 0.5)               │
│  └── 失败码: S2_UNDETECTABLE / S2_POSITION_OFFSET             │
│                                                             │
│  Stage 3: IQA 无参考质量                                     │
│  ├── 工具: BRISQUE + NIQE + AI 伪影检测                      │
│  ├── 检查: 模糊/伪影/LoRA 融合 artifacts                     │
│  └── 失败码: S3_BLUR / S3_ARTIFACT → 调整 LoRA 推理参数       │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、Agent 7: 闭环反馈 📐

### 8.1 职责

解析验证链输出的结构化失败码，分派到对应环节进行调整后重新生成。

### 8.2 失败码分派表

| 失败码 | 含义 | 目标 Agent | 调整动作 |
|:--|:--|:--|:--|
| `S0_BG_UNREALISTIC` | 背景物理不可行 | Agent 3/4 | 重生成背景或调整 ControlNet 种子 |
| `S1_SEMANTIC_MISMATCH` | 语义不一致 | Agent 1 | 调整 LLM prompt 或关键词 |
| `S2_UNDETECTABLE` | 无人机不可检测 | Agent 4 | 调整 LoRA 推理参数（scale/步数） |
| `S2_POSITION_OFFSET` | 位置偏移过大 | Agent 2 | 调整位置编码参数 |
| `S3_BLUR` | 模糊 | Agent 4 | 提高推理步数或降低 CFG scale |
| `S3_ARTIFACT` | 融合伪影 | Agent 3/4 | 调整 ControlNet strength 或 LoRA weight |

### 8.3 闭环策略

```
验证失败 → 记录失败码 → 查询分派表 → 调整目标 Agent 参数 → 重新生成
                                                              │
     ┌────────────────────────────────────────────────────────┘
     │  最多重试 3 次，3 次后标记为 hard_fail 进入人工审查队列
     ▼
  Human-in-the-loop 校准
```

---

## 九、Web 前端 ✅

### 9.1 概述

Flask Web 服务，为整个生成管线提供可视化交互界面。当前已集成 Agent 1（LLM 解析），后续 Agent 2-7 通过 API 逐步对接。**2026-07-30 已完成全功能测试，5 项用例全部通过。**

### 9.2 技术栈

| 层 | 技术 |
|:--|:--|
| 后端 | Flask (Python) |
| 前端 | 原生 HTML/CSS/JS（无框架） |
| LLM | DeepSeek（默认）/ OpenAI / Claude |
| 运行地址 | `http://127.0.0.1:5000` |

### 9.3 三面板布局

```
┌──────────────┬──────────────────────────┬──────────────┐
│  会话记录     │     场景解析器             │  生成图预览   │
│  (320px)     │     (自适应)              │  (420px)     │
│              │                          │              │
│  ┌────────┐  │  ┌────────────────────┐  │  ┌────────┐  │
│  │ #0     │  │  │ Backend: [DeepSeek▼]│  │  │🔴 RGB  │  │
│  │ 城市仰拍 │  │  ├────────────────────┤  │  │ 合成图  │  │
│  │ 15:30  │  │  │                    │  │  │        │  │
│  └────────┘  │  │   JSON 语法高亮     │  │  └────────┘  │
│  ┌────────┐  │  │   预览区域          │  │  ┌────────┐  │
│  │ #1     │  │  │                    │  │  │🔵 IR   │  │
│  │ 森林IR  │  │  ├────────────────────┤  │  │ 热红外  │  │
│  │ 15:31  │  │  │ 示例: 🏙️ 🌲 🏜️ 🌊 │  │  │        │  │
│  └────────┘  │  ├────────────────────┤  │  └────────┘  │
│              │  │ [输入框___________] │  │              │
│              │  │ [解析 →]           │  │              │
│              │  └────────────────────┘  │              │
└──────────────┴──────────────────────────┴──────────────┘
```

### 9.4 配色方案

| 元素 | 颜色 | CSS 变量 |
|:--|:--|:--|
| 主背景 | 白色 | `#ffffff` |
| 面板背景 | 浅灰白 | `#f8fafc` |
| 主色调 | 浅蓝 | `#3b82f6` |
| 辅助色 | 绿色 | `#10b981` |
| 主文字 | 深灰 | `#1e293b` |
| 次文字 | 中灰 | `#64748b` |

### 9.5 API 接口

| 端点 | 方法 | 说明 |
|:--|:--|:--|
| `/` | GET | Web 前端页面 |
| `/api/parse` | POST | 解析自然语言 → JSON + 创建会话 |
| `/api/sessions` | GET | 获取全部会话列表 |
| `/api/session/<id>` | GET | 获取单条会话详情 |
| `/api/session/<id>/images` | POST | 上传 RGB/IR 生成图（后续对接） |

### 9.6 全功能测试记录（2026-07-30）

使用 DeepSeek API + 真实输入 "一架四旋翼无人机在阴天傍晚低空飞近工业区厂房，仰拍观察"。

| # | 测试项 | 结果 | 关键数据 |
|:--|:--|:--|:--|
| 1 | 服务可达 | ✅ HTTP 200 | `curl -s -o /dev/null -w "%{http_code}"` |
| 2 | Dry-run 解析 | ✅ `ok: true` | 9 字段完整返回 |
| 3 | 会话持久化 | ✅ 3 条存储 | `/api/sessions` 全部可查 |
| 4 | 单条回查 | ✅ 正常 | `/api/session/0` 数据完整 |
| 5 | DeepSeek LLM | ✅ 语义正确 | weather=overcast, camera=bottom, action=approach |

#### 9.6.1 DeepSeek API Key 排查过程

初始测试返回 `401 invalid token`，排查发现 `.env` 中 Key 实际有效，根因为 Flask 进程在 Key 更新前启动，使用了旧的环境变量。修复方法：

```bash
kill <pid>                    # 停掉旧 Flask 进程
cd "3-LLM starter" && python3 web_app.py  # 重新启动
```

重启后所有 5 项测试通过。**注意：修改 `.env` 后必须重启 Flask，否则不生效。**

#### 9.6.2 Windows 测试脚本闪退问题

尝试了 3 种方案（bat + `chcp 65001` → bat + `find` → PowerShell + `.bat` 启动器），均闪退。根因为 Windows `curl` 与 bat 转义/中文编码兼容性问题。当前沿用 WSL 端手动测试方案。

### 9.7 会话 JSON 持久化

Web 前端 `sessions: list[dict]` 为内存存储，Flask 重启即丢失。已创建持久化样例文件供参考：

```
3-LLM starter/sessions/sample_session.json
```

样例包含：完整 9 字段 Schema、`_meta` 元数据（输入/时间戳/后端）、`_schema_reference` 速查表（所有字段可选值）。需要全量持久化时，在 `api_parse()` 中加 `json.dump()` 写入 `sessions/` 目录即可。

### 9.8 文件结构

```
3-LLM starter/
├── llm_parser.py            # LLM 解析核心
├── web_app.py               # Flask 服务
├── .env                     # API Key 配置（DeepSeek: sk-b0792...）
├── .env.example             # 配置模板
├── templates/
│   └── index.html           # 前端页面（白底+浅蓝+绿色配色）
├── sessions/
│   └── sample_session.json  # 持久化 JSON 样例（9 字段 + Schema 速查）
└── output_images/           # 生成图片存储（Session ID 命名）
```

### 9.9 启动方式

```bash
cd "3-LLM starter"

# 1. 配置 API Key
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx

# 2. 启动服务
python3 web_app.py
# → 浏览器打开 http://127.0.0.1:5000
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
```

### 10.2 设计原则

| 原则 | 说明 |
|:--|:--|
| 模块化 | 每个 Agent 独立接口，可单独开发/测试/替换 |
| 失败驱动 | 用结构化失败码而非异常，实现精细化闭环 |
| 先跑通主线 | 单机型、单场景优先，验证全链路可行性后再扩展 |
| Human-in-the-loop | 3 次自动重试失败后进入人工审查，而非静默丢弃 |

---

## 十一、当前状态与后续计划

### 11.1 完成项

| 组件 | 状态 | 产出 |
|:--|:--|:--|
| Agent 1: LLM 解析 | ✅ | `llm_parser.py` + 9 字段 Schema + 4 种后端 |
| Web 前端 | ✅ | Flask @ 127.0.0.1:5000，三面板布局 |
| 全功能测试 | ✅ | 5 项全部通过（含 DeepSeek API 真实调用） |
| 会话 JSON 持久化 | ✅ | `sessions/sample_session.json` 参考样例 |
| 无人机 LoRA | ✅ | v2: rank=16, loss=0.0808 |
| API Key | ✅ | DeepSeek sk-b0792... 已配置 `.env` |

### 11.2 待实现

| 组件 | 优先级 | 依赖 |
|:--|:--|:--|
| Agent 2: Transformer 编码 | 高 | Agent 1 ✅ |
| Agent 3: ControlNet 生成 | 高 | Agent 2 |
| Agent 4: LoRA 渲染集成 | 高 | Agent 3 + 无人机 LoRA ✅ |
| Agent 5: IR 转换 | 中 | Agent 4 |
| Agent 6: 验证链 | 中 | Agent 4 + 5 |
| Agent 7: 闭环反馈 | 低 | Agent 6 |
| Web 前端图片上传 | 中 | Agent 4 |

---

## 十二、版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| v1.0 | 2026-07-30 | 初版：7-Agent 架构设计 + Agent 1 完成 + Web 前端上线 |
| v1.1 | 2026-07-30 | 全功能测试：5 项通过、DeepSeek Key 排查、JSON 持久化样例、Windows 脚本闪退记录 |
