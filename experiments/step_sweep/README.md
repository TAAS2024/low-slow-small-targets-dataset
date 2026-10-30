# 局部重绘质量诊断报告（D1–D8 根因 + 步数横向扫描 + 端到端验证）

> 实验日期：2026-10-05　脚本：`run_step_sweep.py` / `run_fix_ab.py` / `downscale_test.py`
> 硬件：单卡 8 GB（VRAM 4552/8188 MiB）、SD1.5 + `drn3_pocket_uav_v3_step2000` LoRA、DDIM、512²
> 结论一句话：**「灰绿糊块 / 沙砾噪声」不是重绘步数不足，而是管线装错了类 + 融合带过宽 + 下采样核振铃。**

---

## 0. TL;DR

| 问题现象 | 真实根因 | 修复后 |
|---|---|---|
| 无人机周围一圈灰绿糊块 | **D1** 用了 txt2img 版 ControlNet 管线，`strength`/`control_image` 被 `**kwargs` 静默吞掉 → 掩膜区域整块重新幻造，且 depth/seg 条件从未生效 | `ring_sat_drop` **−93.9 → −5.0**（pure_sky，18×），`ring_green_shift` **+6.47 → −0.37**（urban） |
| 背景下沉、偏色、被改动一大片 | **D2** 融合带 `feather_px=12` → `dilate 25×25` + `GaussianBlur σ=4.8`，在 64 px 目标上融合带比目标还宽 | `fp = min(feather_px, 5%×目标边长)` → 64 px 目标用 3 px |
| 亮度台阶 | **D3** `_match_lighting` 光斑半径传了「边长」而非「半径」，实际作用半径 76.8 px = 目标半径的 2.4 倍 | 调用改传 `actual_px/2` |
| 目标本身有沙砾感 | **D4** `_composite` 用 `LANCZOS` 做 8× 下采样 → 负瓣振铃 | 倍率 ≥3× 时改用 `BILINEAR`（高频噪声 −36%） |
| overcast/dusty/backlight 场景描述丢失 | **D5** `WEATHER_MAP` 词表比背景池 `_metadata.jsonl` 少 3 项 | 补齐 |

**加步数救不了（实测）**：`core` 区 Immerkær 噪声从 eff≥19 起平在 8.5~8.7（pure_sky）/ 17.1~17.8（urban），**从未优于 steps=0（不重绘）**，且曲线非单调。噪声是背景本身噪声地板（0.49 / 0.32）的 17~56 倍 —— 属于管线引入，不是采样不足。

---

## 1. 目录与产物

| 目录 | 内容 | 状态 |
|---|---|---|
| `full/` | 步数 13 档 × 2 场景 + strength 5 档 + gen_steps 7 档，50 rows | 用**修复前**管线跑（保留，作为缺陷证据） |
| `feather/` | 融合带 2/4/6/8/12/20 px × 2 场景，16 rows | 修复前 |
| `downscale/` | CPU 复刻贴图几何，4 种滤波器 × 4 目标尺寸 | 与 GPU 实测吻合到 2 位小数 |
| `ab_fix/` | 同进程同种子 baseline(txt2img) vs fixed(img2img)，28 rows | 修复后 ✅ |
| `_smoke/` | 早期冒烟 | — |
| `downscale_test.py` | CPU-only，无需 GPU，可随时复现 | — |

关键图：
- `full/sweep_steps_pure_sky.png`、`full/sweep_steps_urban.png`（7272×512，13 档步数条带）
- `ab_fix/ab_fixed_<scene>.png`（2072×512：背景参考 / composite(steps=0) / baseline / fixed）
- `ab_fix/ab_steps_fixed_<scene>.png`（4672×512：修复后 7 档步数）
- `ab_fix/ab_strength_<scene>.png`（3632×512：5 档 strength）

---

## 2. 步数横向扫描（用户指定：0,5,10,…,60，5 步一档）

每个背景只跑一次 `inpaint(skip_cnet=True)` 冻结 `composite / drone_depth / drone_seg / drone_mask / cx / cy / actual_px`，
之后**只改 `num_inference_steps`** 反复调 `_controlnet_repaint`；初始图、条件图、mask、种子、prompt、strength、cnet_scales 全固定。strength=0.65。

### pure_sky（`bg_00ab35f7f4ed.jpg`，1280×851，标签 dawn/dusty）

| steps | eff | core_nz | core_nz×地板 | ring_nz | ring_sat | core_green |
|---|---|---|---|---|---|---|
| 0 | 0 | 8.84 | 17.9 | 2.62 | 182 | −1.27 |
| 5 | 3 | 8.05 | 16.3 | 6.63 | 101 | −2.08 |
| 10 | 6 | 10.48 | 21.2 | 8.14 | 87 | −1.63 |
| 15 | 9 | 9.01 | 18.3 | 7.64 | 88 | −1.65 |
| 20 | 13 | 9.07 | 18.4 | 7.74 | 90 | −1.37 |
| 30 | 19 | 8.66 | 17.6 | 7.89 | 93 | −1.25 |
| 35 | 22 | 8.49 | 17.2 | 7.93 | 93 | −1.03 |
| 45 | 29 | 8.55 | 17.3 | 7.82 | 95 | −0.90 |
| 50 | 32 | 8.68 | 17.6 | 7.93 | 96 | −0.81 |
| 60 | 39 | 8.61 | 17.5 | 7.67 | 94 | −0.67 |

背景噪声地板：`immerkaer=0.49, sigma=1.50, lapvar=19`。区域面积：core=1726 px，ring=5042 px，outside=98020 px。

### urban（标签 morning/clear）

| steps | eff | core_nz | ring_nz | ring_sat | ring_green |
|---|---|---|---|---|---|
| 0 | 0 | 11.67 | 3.15 | 67 | −1.64 |
| 5 | 3 | 12.11 | 11.77 | 91 | **+4.50** |
| 10 | 6 | 16.50 | 14.85 | 77 | +4.17 |
| 15 | 9 | 17.13 | 15.21 | 77 | +4.84 |
| 20 | 13 | 17.82 | 16.11 | 77 | +5.11 |
| 30 | 19 | 17.39 | 15.99 | 77 | +5.60 |
| 40 | 26 | 17.70 | 16.34 | 77 | +5.54 |

**读法**：绿偏在第一次重绘（eff=3）就跳到 +4.5 并**饱和**，之后加步数只会让它更差；`ring_nz` 单调爬升到 15~17 并停住。

### 生成步数对照（pure_sky，重绘固定 15/0.65）

| gen_steps | 10 | 15 | **20（生产）** | 25 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| core_nz | 14.07 | 10.99 | **9.01** | 9.75 | 10.10 | 10.40 | 10.45 |

→ 生产默认 20 已是最优，加多反而更差。

### strength 扫描（修复前）

五个档位（0.30/0.45/0.65/0.80/1.00）输出 PNG **逐字节相同**（md5 全为 `8290e92dcc6727a343dc3f26f14fe10e`）。
→ 这条异常正是发现 D1 的线索。

---

## 3. D1 根因：管线类装错（本报告最重要的发现）

### 证据链

在 installed **diffusers 0.39.0** 中实测签名：

```
StableDiffusionControlNetPipeline.__call__
  (self, prompt, image, height, width, num_inference_steps, timesteps, sigmas,
   guidance_scale, negative_prompt, ..., controlnet_conditioning_scale, guess_mode, ..., **kwargs)
  → has 'image': True | has 'strength': False | has **kwargs: True
  → 源码中 'strength' 出现 0 次；'control_image' 出现 0 次

StableDiffusionControlNetImg2ImgPipeline.__call__
  (self, prompt, image, control_image, height, width, strength,
   num_inference_steps, guidance_scale, negative_prompt, ..., **kwargs)
```

原调用点 `_controlnet_repaint()`：

```python
result = pipe(prompt=..., negative_prompt=BLEND_NEG,
              image=[init_resized, init_resized],        # ← 被当作 ControlNet 条件图
              control_image=[depth_resized, seg_resized],# ← 参数不存在 → 被 **kwargs 吞掉
              strength=strength,                         # ← 参数不存在 → 被 **kwargs 吞掉
              num_inference_steps=steps, ...)
```

**三重失效**：
1. depth/seg 条件图**从未生效**，反而把 composite 照片当成 ControlNet 条件；
2. **没有 img2img 语义** → 每次都从纯噪声重画整张 512²；
3. **`strength` 完全无效**（五档 md5 相同即为铁证）。

代码显然是为 Img2Img 版写的（kwargs 与 Img2Img 签名一一对应），某个时刻类名被换掉，静默破坏语义。

### A/B 实测（`ab_fix/`，同进程同种子，共享同一份 UNet/VAE/ControlNet 权重）

**pure_sky**

| tag | steps | str | eff | sec | core_nz | ring_nz | ring_sat_drop | ring_green | md5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline(txt2img) | 15 | 0.65 | — | 18.58 | 9.01 | 7.64 | **−93.9** | +0.12 | 8b867d6e |
| **fixed(img2img)** | 15 | 0.65 | 9 | **10.36** | **8.43** | **2.83** | **−5.0** | −4.01 | e0e87867 |
| fixed-str | 15 | 0.30 | 4 | 5.20 | 8.35 | 2.64 | +1.7 | −2.10 | 506e0102 |
| fixed-str | 15 | 0.45 | 6 | 7.78 | 9.29 | 2.90 | +0.2 | −2.31 | 1a4c0ccc |
| fixed-str | 15 | 0.80 | 12 | 12.98 | 9.04 | 3.05 | −11.1 | −4.20 | fbfd9055 |
| fixed-str | 15 | 1.00 | 15 | 16.56 | 6.89 | 2.44 | −24.2 | −5.06 | f442d0b0 |
| fixed-steps | 5 | 0.65 | 3 | 4.09 | 8.09 | 2.57 | −2.0 | −3.39 | 3151dca2 |
| fixed-steps | 10 | 0.65 | 6 | 7.19 | 7.78 | 2.48 | −2.7 | −4.20 | 97f20ada |
| fixed-steps | 20 | 0.65 | 13 | 14.56 | 8.52 | 2.80 | −9.0 | −4.09 | 9e807bd0 |
| fixed-steps | 40 | 0.65 | 26 | 27.59 | 8.94 | 2.98 | −10.6 | −4.13 | 96830932 |

**urban**

| tag | steps | str | eff | sec | core_nz | ring_nz | ring_sat_drop | ring_green | md5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline(txt2img) | 15 | 0.65 | — | 16.35 | 17.13 | 15.21 | +10.3 | **+6.47** | 08ecff7b |
| **fixed(img2img)** | 15 | 0.65 | 9 | **9.97** | **9.37** | **3.04** | +3.7 | **−0.37** | 889c3dcb |
| fixed-str | 15 | 0.30 | 4 | 5.31 | 9.95 | 3.11 | +1.6 | −0.78 | d29e9e8e |
| fixed-str | 15 | 0.45 | 6 | 7.85 | 9.34 | 3.00 | +2.1 | −0.97 | 5fa9db11 |
| fixed-str | 15 | 0.80 | 12 | 13.28 | 12.13 | 3.75 | +0.6 | −0.76 | 1289069a |
| fixed-str | 15 | 1.00 | 15 | 17.46 | 9.00 | 3.27 | −6.5 | −0.97 | 3be49569 |
| fixed-steps | 5 | 0.65 | 3 | 4.39 | 9.50 | 3.01 | +1.8 | −0.70 | 53ae3084 |
| fixed-steps | 20 | 0.65 | 13 | 13.75 | 9.25 | 2.88 | +4.5 | −0.26 | 4cbaf64c |
| fixed-steps | 40 | 0.65 | 26 | 27.11 | 10.27 | 3.19 | +6.1 | −0.28 | a63d4c0f |

**三项硬结论**
1. `fixed-str` 五档 strength → **5 种不同 md5**（两场景均 5/5；baseline 为 1 种）→ D1 诊断决定性证明，`strength` 恢复生效。
2. `ring_sat_drop`：pure_sky **−93.9 → −5.0**（18× 改善）；`ring_green_shift`：urban **+6.47 → −0.37**（灰绿晕染消失）。
3. 耗时 **18.6 s → 10.4 s**（1.8× 加速）：真实执行 9 步而非 15 步，`effective_denoise_steps` 从「虚构」变为真实。

**附加观察**：`strength=1.00` 在 img2img 语义下等价于「从纯噪声起步」，即**复现了修复前的行为**（core_nz 最低 6.89，但背景下沉最严重 −24.2）。这解释了历史数据集为什么看起来「能用」——它一直相当于 strength=1.0 的整块重画。

---

## 4. D2 融合带宽度（`feather/`，16 rows）

`ring_green_shift` 与 `ring_nz` **单调随 feather 增长**：

| scene | feather 2 | 4 | 6 | 8 | **12（生产）** | 20 |
|---|---|---|---|---|---|---|
| pure_sky ring_contam% | 74.1 | 82.4 | 90.9 | 96.4 | 99.6 | 100.0 |
| pure_sky sat_drop | **−1.0** | −19.8 | −39.6 | −59.2 | **−93.9** | −120.5 |
| pure_sky green_shift | +0.01 | +0.04 | +0.06 | +0.06 | +0.12 | +0.18 |
| pure_sky ring_nz | 3.01 | 3.85 | 4.84 | 5.85 | 7.64 | 9.08 |
| urban green_shift | **+0.82** | +2.04 | +3.27 | +4.51 | **+6.47** | +7.77 |
| urban ring_nz | 5.64 | 7.61 | 9.72 | 11.86 | 15.21 | 17.42 |
| urban core_nz | 16.51 | 17.02 | 17.06 | 17.12 | 17.13 | 17.13 |

feather=2 px 时 urban 绿偏仅 +0.82 —— 是生产 12 px 的 **1/8**。
`ring_contam` 即便 feather=2 仍 >74%，那是 D3（光照匹配整片背景乘标量）造成的**纯亮度**改动，不改变饱和度/色相。

---

## 5. D4 下采样核（`downscale_test.py`，CPU-only，可复现）

复刻 `_composite` 贴图几何、只参数化 resample；与 GPU 实测吻合到 2 位小数（LANCZOS core_immerkaer 8.840 vs 扫描实测 8.84 ✅）。

| 滤波器 | tp32 | tp64 | tp96 | tp128 |
|---|---|---|---|---|
| LANCZOS（原） | 12.11 | 8.84 | 6.85 | 5.47 |
| BICUBIC | 11.56 | 8.11 | 6.17 | 4.81 |
| **BILINEAR** | **7.62** | **5.69** | **4.40** | **3.49** |
| BOX(area) | 13.37 | 8.83 | 6.82 | 5.27 |
| NEAREST | 28.85 | 18.80 | 14.16 | 10.01 |

**BILINEAR 在所有目标尺寸下降高频 ~36%（lapvar −43~47%）**；LANCZOS/BICUBIC 带负瓣，8× 下采样产生振铃/摩尔纹。

---

## 6. 修复清单（全部已落地，两份副本 md5 一致）

| # | 文件 | 位置 | 修法 |
|---|---|---|---|
| D1 | `lora_inpainter_v7.py` ×2 | `load()` / `_get_nolora_pipe()` / `_controlnet_repaint()` | `StableDiffusionControlNetPipeline` → `...Img2ImgPipeline`；调用点按管线类型动态传参（img2img：`image=init_resized` + `control_image=[depth,seg]` + `strength`），否则打印告警走旧路径 |
| D2 | 同上 | `_controlnet_repaint()` 融合带 | `fp = int(max(2, min(int(feather_px), round(0.05*actual_px))))`；`GaussianBlur σ = max(0.8, fp/2.5)` |
| D3 | 同上 | `_match_lighting` 调用点 | `actual_px` → `actual_px / 2.0`（半径语义） |
| D4 | 同上 | `_composite()` | `_ds_ratio = ds/max(nw,nh)`；`≥3.0` 用 `Image.BILINEAR`，否则 `LANCZOS` |
| D5 | 同上 | `WEATHER_MAP` | 补 `overcast` / `dusty` / `backlight` |
| **D6** | `condition_generator_v7.py` ×2、`condition_generator.py` | `_get_midas()` | `MidasDetector.from_pretrained("lllyasviel/Annotators")` 在 proxy 未启动 / hf-mirror 抖动时抛 `ConnectionError`，**整条生成管线崩掉**；模型其实已在 HF cache 内。加 `try/except` 降级 `local_files_only=True` |
| **D7** | `web_app.py` ×2 | `_apply_generation()` | 调用方会传 session entry（regenerate/focus_drone）**或**裸 images dict（parse）。原实现只判断 `"images" not in session` → 给裸 images dict 塞了个**嵌套** `images` 键，外层十槽位全空 |
| **D8** | `web_app.py` ×2 | session entry | `warnings` 只出现在 HTTP 响应、不落 session，事后无法排查 |

`md5 = 7b2e6f0a9463e3bd476504b56cb1dab5`（36,778 字符），`py_compile` 通过，`0-workspace/generator/` 与 `5-Controlnet/` 两份一致。

**保持不变**：生产默认 `blend_steps=15`、`blend_strength=0.65`（→ 真实 9 步）。
A/B 显示 eff 4~13 区间内 core_nz 无显著差异（8.35~9.29），0.65 是「背景保真 vs 无人机精化」的合理折中；
若追求背景绝对不变，可下调到 0.45（pure_sky sat_drop +0.2）。

---

## 6.5 前端 / 服务端端到端验证（`3-LLM starter/e2e_webcheck.py`）

GPU 释放后补做的整链路验证（此前因 8 GB 卡跑不了两份 SD1.5 而挂起）。
脚本自带启动/关机、日志扫描、20 项断言，产物 `_e2e_report.json` + `_e2e_stdout.log`。

**首轮跑出的真实回归 —— D7（严重）**

| 断言 | 首轮结果 | 根因 |
|---|---|---|
| 十槽位已填充 | **1/10**（仅 `bg`） | `_apply_generation(images, ...)`：调用方传裸 images dict，函数内 `if "images" not in session` 成立 → 塞进一个**嵌套** `images` 键，外层槽位全空 |
| `/api/regenerate` | `无背景图，无法重置` | 级联自 D7（`session["images"]["bg"]` 为空） |
| `/api/focus_drone` | HTTP 400 `请先生成场景（无背景图）` | 级联自 D7 |

`/api/parse` 走的是 `_apply_generation(images, ...)`，而 `/api/regenerate`、`/api/focus_drone` 走 `(session, ...)` —— 只有 parse 这条路径坏，所以**前端的「一键生成」结果区会全空**，而单点按钮报错。
修复后十槽位 `filled=10/10`、10 个 URL 全部 HTTP 200，两个下游端点恢复正常。

**其它首轮暴露的问题**

- `/api/session/<sid>/images` 期望 **multipart 表单文件**，不是 JSON 路径字符串（原测试发 JSON → `saved={}`）；已按真实契约改用 multipart 验证通过。
- `/api/shutdown` 在回包前就 `SIGTERM` 自己 → 客户端必然看到连接被断，属预期行为，断言应容忍。
- `/api/regenerate` **必须**带 `slot`（前端一直带），缺失时正确返回 400 `未知 slot: ''`。

**最终结果（末轮复测，端口 5078）**

```
20/20 PASS
  /api/health                         ok
  /api/parse 全管线(skip_cnet=false)   HTTP 200 100.0s  十槽位 10/10  URL 10/10 200  warnings=clean
  /api/regenerate 并发 ×2              两条均 ok，无 CUDA 错误；wall=58.6s = max(29.5,58.6)
  /api/focus_drone                    HTTP 200 14.0s    十槽位齐全
  /api/session/<sid>/images (multipart) saved=['final_rgb','ir']
  /api/session/<sid> 读回               images_filled=10/10
  /api/sessions                        n=1
  /api/shutdown + 进程退出             code=-15
  服务端日志                           无 Traceback / CUDA error
  网络相关日志行数                      0（含 hf-mirror / Retrying / MaxRetryError）
```

最后一行值得单列：**整条生成管线现在零网络依赖**。此前单是深度估计器的 revision 检查就要空转 23 s 并可能直接崩掉（见 §6.6）；
同时 `/api/parse` 耗时 123.9s → 100.0s 也主要来自这里。

**并发串行化判据的坑（记录备查）**：两个并发请求的耗时是 `[29.5s, 58.6s]`，`wall=58.6s`。
`sum=88.2s` 把「排队等待」算了两遍，用 `wall >= 0.85*sum` 判断会**误报失败**。
正确判据：`wall ≈ max(t)` 且 `max(t) >= 1.5*min(t)`（排队者耗时≈2× 单人耗时）→ 已改为该口径。

**会话存储单元测试（无需 GPU）**

| 检查 | 结果 |
|---|---|
| `MAX_SESSIONS=200`，put 260 条后 len==200 | PASS |
| 最旧会话被淘汰、最新可读 | PASS |
| `_alloc_sid` 500 次单调不重复 | PASS |
| 8 线程 × 300 次并发 alloc/put/list 无异常 | PASS |
| `_apply_generation` 裸 images dict 形态回填 | PASS（D7 回归守卫） |
| `_apply_generation` 完整 session 形态回填 | PASS |

---

## 6.6 D6 实测：深度估计器的网络依赖（复现 + 修复验证）

```
# 修复前（E2E 首轮）
  hf-mirror.com HEAD → ProxyError(127.0.0.1:7890 refused) → Retrying 1/5 … 5/5
  ConnectionRefusedError → 抛到 /api/parse → 只回一张背景图

# 修复后（本机 http_proxy 指向死端口强制离线）
  ⚠ MidasDetector 本地缓存未命中（...），联网下载 ...   ← 仅在缓存缺失时才联网
  OK MidasDetector  6.1s            （此前 29.7s：23s 全花在无效重试）
```

⚠️ **重要**：模型权重其实早已在 `~/.cache/huggingface/hub/models--lllyasviel--Annotators` 内。
即「联网只是为了查 revision」，网络不可达时整条生成管线却会崩 —— 这是纯粹的自伤。
现已改为 `local_files_only=True` 优先：命中缓存零网络、启动更快、**权重版本可复现**（研究产物本就不该跟随上游更新）。

---

## 7. 🔴 对论文口径的影响（必须处理）

历史数据集（`out_D` 等）是在 D1 缺陷下产出的，因此：

1. 论文若写「ControlNet 双条件（depth+seg）局部重绘」——**此前不成立**（条件图从未被消费）。修好后成立。
2. 论文若写「strength=0.65 → 9 个有效去噪步」——**此前不成立**（该参数被忽略，实际等价 strength=1.0 的整块重画）。修好后成立。
3. 论文若依赖「合成后精化保留背景纹理」——此前不成立（背景在掩膜内被整块重新幻造）。
4. 但**管线第 1~4 步（LoRA 生成 → rembg 抠图 → alpha 贴入 → 注入 depth/seg）完全真实且未变**；采集的 `drone_depth/drone_seg/drone_mask` 也真实存在，只是第 5 步没消费它们。

**三条可选路线**（需 Tiger 拍板）：
- (a) 重跑数据集（成本：单张 ≈10 s CN 重绘 + 生成，按现有产能可接受）；
- (b) 保留旧数据但在论文中明确声明第 5 步的缺陷与修复，并加一组修复后的小规模对照；
- (c) 论文改写第 5 步描述，使其与实际行为一致（削弱方法贡献，不推荐）。

---

## 8. 复现命令

```bash
cd experiments/step_sweep
python3 -u run_step_sweep.py --scenes pure_sky urban --out full            # 步数/strength/gen_steps 扫描
python3 -u run_step_sweep.py --feather-only --feathers 2 4 6 8 12 20 --out feather
python3    downscale_test.py --out downscale                              # CPU-only
python3 -u run_fix_ab.py --scenes pure_sky urban --out ab_fix             # 修复前后 A/B
```

```bash
cd "3-LLM starter"
python3 -u e2e_webcheck.py --port 5077        # 端到端 20 断言（需 GPU，~7 min，自动起停服务）
```

---

## 9. 未闭环 / 待办

- [x] web_app 端到端验证（`/api/health`、`/api/parse`、`/api/regenerate`、`/api/focus_drone`、session 十槽位）—— **20/20 PASS**（见 §6.5）
- [ ] 论文口径修正 / 数据集重跑决策（§7）
- [ ] `BG_MATCH=labels` 默认是否启用 —— **注意：当前代码已启用**（日志 `[background_searcher] match=tod+weather cand=669/2695`），会改变数据集语义与论文「随机采样 37,852」叙述，需在论文中改写或说明
- [ ] `mountain` 场景 ~300 张图缺元数据（覆盖 99.2%）
- [ ] 视觉复核：`ab_fix/ab_fixed_*.png` 需人眼确认（本报告只给量化指标）
- [ ] 前端未暴露 `cnet_scales` / `use_lora`（后端已支持）；`blend_strength` 已于本轮加入面板并实时显示「≈N 有效步」
- [ ] `/api/search-background`、`/api/generate-lora`、`/api/generate-conditions` 三个端点为前端死代码（保留供下游 Agent/legacy 使用），未做冒烟验证

## 10. 本轮改动文件清单

| 文件 | md5 | 说明 |
|---|---|---|
| `0-workspace/generator/lora_inpainter_v7.py`、`5-Controlnet/lora_inpainter_v7.py` | `7b2e6f0a9463e3bd476504b56cb1dab5` | D1–D5（36,778 字符） |
| `3-LLM starter/condition_generator_v7.py`、`0-workspace/app/condition_generator_v7.py` | `9a057c4d76f970e923761b47e96798c7` | GPU 串行化 + D6 |
| `3-LLM starter/condition_generator.py` | — | GPU 串行化 + D6 |
| `3-LLM starter/web_app.py` (`55bd6887…`)、`0-workspace/app/web_app.py` (`b9ff1b16…`) | 见左 | D7 + D8；两份有意不同（活跃版多 legacy 端点） |
| `3-LLM starter/templates/index.html`、`0-workspace/app/templates/index.html` | `9028f0318a37a973e2f0b240c6ba6408` | 重绘强度输入 + 实时有效步数 |
| `3-LLM starter/e2e_webcheck.py` | 新建 | 端到端验证（20 断言） |
| `experiments/step_sweep/{run_step_sweep,run_fix_ab,downscale_test}.py` | — | 实验脚本 |
| `experiments/step_sweep/README.md` | 新建 | 本报告 |

