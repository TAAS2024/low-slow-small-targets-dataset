---
tags: [输入端预校验, Validator, S1, JSON校验, LLM]
stage: design
status: 设计完成待编码
version: v1.0
created: 2026-08-05
depends: [[../·重点节点总结笔记/6-Validator设计与搭建]], [[../·重点节点总结笔记/7-持续学习循环设计]]
---

# S1 — Agent 1 (LLM) JSON 输出校验

> 输入端预校验第一关。纯规则引擎，零 GPU 消耗。
> 对应任务规划 [[../·重点节点总结笔记/6-Validator设计与搭建#S1：Agent 1 (LLM) JSON 输出校验|S1 设计]]。

---

## 一、定位

```
输入:  Agent 1 产出的 9 字段结构化 JSON
输出:  S1Verdict ∈ {PASS, FAIL}
```

**审谁**：Agent 1 产出的 9 字段结构化 JSON（drone_type, trajectory, time_of_day, weather, scene_type, scene_description, modality, camera, confidence_note）。

**为什么**：LLM 有能力输出逻辑矛盾的组合（如 `time_of_day="night"` + `weather="backlight"`）。如果不拦截，矛盾会污染整个下游管线——Agent 2 基于矛盾的场景描述做空间编码、Agent 3/4 生成与 JSON 元数据冲突的图像，最终在 S7 才被光照一致性检查发现——浪费一整轮 GPU 推理。

---

## 二、判定逻辑

```
Schema 不完整          → FAIL (A1_SCHEMA_INCOMPLETE)
枚举值不合法            → FAIL (A1_ENUM_INVALID)
场景逻辑矛盾            → FAIL (A1_LOGIC_CONFLICT)
description 与字段不一致 → FAIL (A1_DESC_MISMATCH)
trajectory 值域异常     → FAIL (A1_TRAJ_INVALID)
camera 参数越界         → FAIL (A1_CAMERA_INVALID)
全部通过               → PASS
```

> ⚠️ 短路求值：任一检查失败立即返回 FAIL，不继续后续检查。

---

## 三、检查项详表

| # | 检查项 | 方法 | 失败码 |
|:--|:--|:--|:--|
| 1 | Schema 完整性 | 9 字段齐全 + trajectory 非空 + camera 含必填子字段 | `A1_SCHEMA_INCOMPLETE` |
| 2 | 枚举值合法性 | drone_type / weather / time_of_day / scene_type / camera.position 均在预定义集合内 | `A1_ENUM_INVALID` |
| 3 | 场景逻辑自洽 | 互斥规则表（见 §四） | `A1_LOGIC_CONFLICT` |
| 4 | scene_description 关键词交叉检查 | description 中出现 "bright sun" 但 weather="overcast" → 不一致 | `A1_DESC_MISMATCH` |
| 5 | trajectory 值域 | 所有 distance > 0，action 枚举合法，t 递增 | `A1_TRAJ_INVALID` |
| 6 | camera 参数合理区间 | elevation_deg 在相机位置对应的合理范围、fov_deg ∈ [20, 120] | `A1_CAMERA_INVALID` |

---

## 四、互斥规则表

| 组合 | 判定 | 理由 |
|:--|:--|:--|
| `time_of_day=night` + `weather=backlight` | ❌ 拒绝 | 逆光需要太阳光源，夜晚不存在 |
| `time_of_day=noon` + `weather=fog` | ⚠️ 警告 | 正午辐射强，浓雾罕见但可能 |
| `scene_type=indoor` + `weather=rainy` | ❌ 拒绝 | 室内不下雨 |
| `camera.position=bottom` + `elevation_deg>0` | ❌ 拒绝 | 底部视角不可能仰拍 |

> 规则表持续扩充，存放于 `config/exclusion_rules.yaml`。

---

## 五、依赖

- `config/ENUM_REGISTRY.yaml` — 所有枚举值集合的单一事实来源
- `config/exclusion_rules.yaml` — 互斥规则表
- Agent 1 输出的 JSON 文件路径（由 pipeline 传入）

---

## 六、代码结构

```
S1 Agent 1 JSON校验/
├── S1-JSON校验Validator.md    ← 本文件
├── code/
│   └── s1_json_validator.py   ← 主校验逻辑 (~80 行)
├── config/
│   ├── ENUM_REGISTRY.yaml     ← 枚举值注册表
│   └── exclusion_rules.yaml   ← 互斥规则表
└── tests/
    └── test_s1_json.py        ← 单元测试
```

---

## 七、实现要点

- 纯 JSON 解析 + 字符串匹配，约 80 行
- 互斥规则表用 YAML 维护，方便非工程师扩充
- 描述关键词交叉检查用简单的子串匹配（不需要 NLP）
- 与下游 S2 完全解耦——S1 FAIL 时直接打回 Agent 1 重新生成
