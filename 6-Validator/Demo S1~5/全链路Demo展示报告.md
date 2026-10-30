# 6-Validator Full-Pipeline Demo Report

> Generated: 2026-08-05 | Total Scenarios: 48 | PASS: 23 | FAIL: 22 | ⚠️ WARN: 3

---

## Architecture

```
LLM JSON ──→ S1 ──→ S2 ──→ S3 ──→ S4 ──→ S5 ──→ [Generation Pipeline] ──→ S6 ──→ S7 ──→ S8 ──→ S9
 (Agent 1)   JSON  Transf  CtrlN   LoRA    IR       RGB Synthesis          Quality  Consis  Traj   Detection
             Valid ormer    Valid   Valid   Valid                           Valid   tency  Valid   Valid
             (Gate) (Gate)  (Gate)  (Gate)  (Warn)                         (Audit) (Audit) (Audit) (Audit)
```

**S1-S5**: Input pre-validation (gates — halt dataflow on failure)  
**S6-S9**: Output post-audit (reviews generated results)

---

## Summary

| Validator | Scenarios | ✅ PASS | ❌ FAIL | ⚠️ WARN |
|:--|:--:|:--:|:--:|:--:|
| S1 JSON Schema | 14 | 6 | 8 | — |
| S2 Transformer | 12 | 6 | 6 | — |
| S3 ControlNet | 8 | 4 | 4 | — |
| S4 LoRA | 7 | 3 | 4 | — |
| S5 IR | 7 | 4 | **0** | 3 |
| **Total** | **48** | **23** | **22** | **3** |

---

## S1 — JSON Schema Validation (6 checks)

**Gate type**: Hard fail (short-circuit — first FAIL halts all subsequent)

| # | Check | Failure Code | Scenarios |
|:--|:--|:--|:--|
| Chk-1 | Schema Completeness | `A1_SCHEMA_INCOMPLETE` | ✅ 9 fields complete / ❌ missing 'weather' / ❌ trajectory empty |
| Chk-2 | Enum Validity | `A1_ENUM_INVALID` | ✅ legal values / ❌ weather='tornado' / ❌ drone='spaceship' |
| Chk-3 | Logic Conflicts | `A1_LOGIC_CONFLICT` | ✅ morning+clear / ❌ night+backlight / ❌ indoor+rainy |
| Chk-4 | Description Cross-check | `A1_DESCRIPTION_BRIEF` | ✅ desc says 'quadcopter' / ❌ desc says 'rainy' but weather='clear' |
| Chk-5 | Trajectory Range | `A1_TRAJECTORY_SPARSE` | ✅ 3 pts, t monotonic, actions legal / ❌ distance<0 / ❌ t non-monotonic |
| Chk-6 | Camera Parameters | `A1_CAMERA_INCONSISTENT` | ✅ fov=90 legal / ❌ fov=200 / ❌ top_down but elevation=10° |

**Demo images**:
- ![S1 PASS](Demo%20S1~5/demo_s1_pass.png)
- ![S1 FAIL](Demo%20S1~5/demo_s1_fail.png)

**S1 FAIL example output**:
```
Result: ❌ FAIL [A1_SCHEMA_INCOMPLETE]
Reason: Missing field: 'weather'
Note: Chk-2 through Chk-6 short-circuited (not reached)
```

---

## S2 — Transformer Bbox Validation (6 checks)

**Gate type**: Hard fail (per-frame check, first frame FAIL halts all)

| # | Check | Failure Code | Scenarios |
|:--|:--|:--|:--|
| Chk-1 | Bbox In-Bounds | `A2_BBOX_OOB` | ✅ 8 frames within 256×256 / ❌ x=-10 left edge / ❌ y+h=300 bottom edge |
| Chk-2 | Bbox Size | `A2_SIZE_ANOMALY` | ✅ 40×40 reasonable / ❌ 2×2 too small / ❌ 256×256 too large |
| Chk-3 | Position Continuity | `A2_POSITION_JUMP` | ✅ smooth trajectory / ❌ sudden 200px jump between frames |
| Chk-4 | Seg Non-Empty | `A2_SEG_EMPTY` | ✅ seg contains drone class (2) / ❌ seg all zeros |
| Chk-5 | Depth Validity | `A2_DEPTH_FLAT` | ✅ depth ∈ [0,1] float / ❌ all -1 (invalid) |
| Chk-6 | Frame Count | `A2_FRAME_COUNT_MISMATCH` | ✅ 8 frames ↔ traj_len=8 / ❌ 8 frames ↔ traj_len=3 |

**Demo images**:
- ![S2 PASS](Demo%20S1~5/demo_s2_pass.png)
- ![S2 FAIL](Demo%20S1~5/demo_s2_fail.png)

**S2 FAIL example output**:
```
Result: ❌ FAIL [A2_BBOX_OOB]
Reason: Frame 3: bbox x=-10 exceeds left canvas boundary
Note: Subsequent checks short-circuited
```

---

## S3 — ControlNet Condition Map Validation (4 checks)

**Gate type**: Hard fail

| # | Check | Failure Code | Scenarios |
|:--|:--|:--|:--|
| Chk-1 | Size Consistency | `A3_SIZE_MISMATCH` | ✅ all 256×256 / ❌ depth=128×128 |
| Chk-2 | Seg-Bbox Alignment | `A3_ALIGNMENT` | ✅ centroid ≈ bbox center / ❌ deviation > threshold |
| Chk-3 | Seg Boundary Smoothness | `A3_SEG_JAGGED` | ✅ solid mask dilation < 50% / ❌ checkerboard dilation > 50% |
| Chk-4 | Depth Consistency | `A3_DEPTH_FLAT` | ✅ drone depth < background / ❌ all pixels = 0.5 |

**Demo images**:
- ![S3 PASS](Demo%20S1~5/demo_s3_pass.png)
- ![S3 FAIL](Demo%20S1~5/demo_s3_fail.png)

**S3 FAIL example output**:
```
Result: ❌ FAIL [A3_SEG_JAGGED]
Reason: Checkerboard seg: dilation inflation 78% > 50% threshold
```

---

## S4 — LoRA Output Validation (3 checks)

**Gate type**: Hard fail

| # | Check | Failure Code | Scenarios |
|:--|:--|:--|:--|
| Chk-1 | Concept Bleed | `A4_CONCEPT_BLEED` | ✅ background ~0.47 vs drone ~0.18 / ❌ uniform texture → drone leaked |
| Chk-2 | Texture Diversity | `A4_TEXTURE_REPEAT` | ✅ 3 frames distinct / ❌ 3 identical frames (overfit) |
| Chk-3 | Color Cast | `A4_COLOR_CAST` | ✅ normal range / ❌ whole frame dark ~0.05 / ❌ red channel spike |

**Demo images**:
- ![S4 PASS](Demo%20S1~5/demo_s4_pass.png)
- ![S4 FAIL](Demo%20S1~5/demo_s4_fail.png)

**S4 FAIL example output**:
```
Result: ❌ FAIL [A4_CONCEPT_BLEED]
Reason: Frame SSIM=0.98 exceeds threshold → drone texture contaminates background
```

---

## S5 — IR Conversion Validation (3 checks) ⚠️ Special

**Gate type**: ⚠️ **Warn-only, never halts** (defensive pass — pass is always True)

| # | Check | Warning Code | Scenarios |
|:--|:--|:--|:--|
| Chk-1 | Size Match | `A5_SIZE_MISMATCH` | ✅ 256×256 ↔ 256×256 / ⚠️ 128×128 ≠ 256×256 |
| Chk-2 | Grayscale Confirm | `A5_NOT_GRAYSCALE` | ✅ single-channel (H,W) / ⚠️ 3-channel pseudo-color, std=62.36 |
| Chk-3 | Histogram Reasonable | `A5_HISTOGRAM_ANOMALY` | ✅ bimodal (sky~60, ground~180) / ⚠️ unimodal, all pixels≈100 |

**Demo images**:
- ![S5 PASS](Demo%20S1~5/demo_s5_pass.png)
- ![S5 WARN](Demo%20S1~5/demo_s5_warn.png)
- ![S5 ALL WARN](Demo%20S1~5/demo_s5_allwarn.png)

**S5 warning output (defensive pass)**:
```
Result: ⚠️  PASS (defensive)
pass = True          ← always True
failure_code = None  ← always None
3 warnings issued — but dataflow is NOT halted
```

### S5 Design Issue & Proposed Fix

**Current behavior**: IR = code_convert(RGB) is treated as a "deterministic transform with no independent failure mode." All checks are warn-only.

**Design gaps**:
1. Deterministic code can still be buggy (e.g., swapped resize params causing size mismatch)
2. Upstream edge cases can punch through (e.g., code_convert producing pseudo-color for some inputs)
3. Always returning PASS makes this Validator functionally useless

**Proposed fix ("Two Hard, One Soft")**:
| Check | Current | Proposed | Rationale |
|:--|:--|:--|:--|
| Chk-1 SIZE_MISMATCH | WARN | **FAIL** | Wrong dimensions → downstream impossible, must halt |
| Chk-2 NOT_GRAYSCALE | WARN | **FAIL** | Pseudo-color IR → pipeline guaranteed error, must halt |
| Chk-3 HISTOGRAM_ANOMALY | WARN | **WARN** | Keep as warning (false positive risk: all-sky/fog scenes) |

---

## Visual Demo File List

| File | Validator | Verdict | Content |
|:--|:--|:--|:--|
| `demo_s1_pass.png` | S1 JSON | PASS | Complete legal JSON, 6/6 checks pass |
| `demo_s1_fail.png` | S1 JSON | FAIL | Missing 'weather' field → short-circuited |
| `demo_s2_pass.png` | S2 Transformer | PASS | 8 frames, drone+bbox all within canvas |
| `demo_s2_fail.png` | S2 Transformer | FAIL | F3 bbox x=-10 off left edge |
| `demo_s3_pass.png` | S3 ControlNet | PASS | Clean seg overlay + gradient depth |
| `demo_s3_fail.png` | S3 ControlNet | FAIL | Checkerboard seg → jagged boundaries |
| `demo_s4_pass.png` | S4 LoRA | PASS | 3 frames, distinct drone textures |
| `demo_s4_fail.png` | S4 LoRA | FAIL | Uniform texture → concept bleed |
| `demo_s5_pass.png` | S5 IR | PASS | 256×256 grayscale IR, bimodal histogram |
| `demo_s5_warn.png` | S5 IR | WARN | 128×128 pseudo-color → warnings, still PASS |
| `demo_s5_allwarn.png` | S5 IR | WARN | 🔑 All 3 warnings firing — pass remains True |

### Corresponding S6-S9 Demos (existing)
Located in `Demo S6~9/`:
- `demo_pass.png` — Full pipeline PASS → Training Pool
- `demo_s6_fail.png` — S6 Quality halt (blur + noise)
- `demo_s7_fail.png` — S7 Consistency halt (size anomaly)
- `demo_s8_fail.png` — S8 Trajectory halt (position jump)
- `demo_s9_fail.png` — S9 Detection halt (YOLO miss)

---

## Suggested Presentation Order

1. **S1-S4 Normal Gates**: Show any FAIL image to demonstrate "anomaly detected → halt → downstream skipped" short-circuit logic
2. **S5 Defensive Pass**: Show `demo_s5_allwarn.png` — "3 alarms all firing, yet pass is True"
3. **Introduce Fix**: "Two Hard, One Soft" — promote SIZE_MISMATCH and NOT_GRAYSCALE to FAIL
4. **S6-S9 Output Audit**: Demonstrate that post-generation pipeline also has full validation chain
