"""
低慢小数据集生成架构 — LLM Parser Web 服务 (v7.2)
=================================================
本地 Web 前端 + API，双 JSON 输出模式：
  - background_spec.json — Agent 2 背景匹配
  - full_scene_spec.json  — Agent 3/4/5/6 全管线

启动: python web_app.py
访问: http://127.0.0.1:5000
"""

import os
import sys
import json
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # → 0-workspace
from lss_paths import OUTPUT_DIR as IMAGE_DIR  # noqa: E402  (single source of truth)

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

from llm_parser import parse, parse_to_dual_json
from background_searcher import search_background, get_placeholder_image

# 加载 .env（优先本目录；原项目 3-LLM starter/ 与归档 app/ 均适用）
load_dotenv(Path(__file__).parent / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ============================================================
# 会话存储（内存态，重启丢失）
# ============================================================
# v7.2 修复：原先用 list + len(sessions) 当 id，一旦裁剪/删除就会 id 冲突；
# 且列表无上限会随使用无限增长。改为 dict 按 sid 索引 + 单调计数器 + 容量裁剪。
# Flask 的 app.run() 默认 threaded=True，并发请求会同时触碰本结构，故加锁。
_LOCK = threading.RLock()
sessions: dict = {}
_SID_SEQ = 0
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "200"))


def _alloc_sid() -> int:
    global _SID_SEQ
    with _LOCK:
        sid = _SID_SEQ
        _SID_SEQ += 1
        return sid


def _put_session(entry: dict) -> None:
    """写入会话，并按 MAX_SESSIONS 裁掉最旧的若干条。"""
    with _LOCK:
        sessions[entry["id"]] = entry
        while len(sessions) > MAX_SESSIONS:
            sessions.pop(min(sessions), None)


def _session_list() -> list:
    with _LOCK:
        return [sessions[k] for k in sorted(sessions)]


# 图片输出目录（IMAGE_DIR 由 lss_paths 提供，导入时已确保存在）

# 可透传给生成管线的旋钮（前端/curl 均可覆盖）
_KNOB_INT = ("gen_steps", "blend_steps")
_KNOB_FLOAT = ("blend_strength",)


def _knobs(data: dict, allowed: tuple) -> dict:
    """
    从请求体提取可选管线旋钮，全部带安全默认值。

    allowed 决定哪些键被接受，例如：
        _knobs(data, ("skip_cnet", "use_lora", "blend_steps", "blend_strength"))
    """
    data = data or {}
    out = {}
    if "skip_cnet" in allowed:
        out["skip_cnet"] = bool(data.get("skip_cnet", False))
    if "use_lora" in allowed:
        out["use_lora"] = bool(data.get("use_lora", True))
    for name in _KNOB_INT:
        if name in allowed and data.get(name) is not None:
            out[name] = max(1, int(data[name]))
    for name in _KNOB_FLOAT:
        if name in allowed and data.get(name) is not None:
            out[name] = float(data[name])
    if "cnet_scales" in allowed and data.get("cnet_scales") is not None:
        sc = data["cnet_scales"]
        if isinstance(sc, (list, tuple)) and len(sc) == 2:
            out["cnet_scales"] = (float(sc[0]), float(sc[1]))
    return out


def _init_session_images(sid: int):
    """初始化会话的 10 图存储结构（v7.2: +ir, +bbox）。"""
    return {
        "bg": [],          # 背景图（单张）
        "bg_depth": [],    # 背景深度图（单张）
        "bg_seg": [],      # 背景分割图（单张）
        "drone_depth": [], # 无人机深度图（多帧）
        "drone_seg": [],   # 无人机分割图（多帧）
        "final_rgb": [],   # 最终合成图（多帧）
        "ir": [],          # 白热红外图（多帧）
        "bbox": [],        # bbox 可视化（多帧）
        "drone_clean": [], # LoRA 抠图后缩放前（多帧）— 备选栏
        "drone_raw": [],   # LoRA 原始生成图 512² 白底（多帧）— 备选栏
    }


# 前端可展示的槽位（顺序即写入顺序）
_ALL_SLOTS = ("bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg",
              "final_rgb", "ir", "bbox", "drone_clean", "drone_raw")


def _apply_generation(session: dict, sid: int, gen_result: dict) -> None:
    """把一次生成的产物回填到会话的 images 里。

    v7.3: 调用方会传两种形态 —— 完整 session entry（/api/regenerate、/api/focus_drone）
    与裸 images dict（/api/parse）。原实现只判断 `"images" not in session`，
    当传入裸 images dict 时会往里塞一个**嵌套的** `images` 键，外层十个槽位全部为空。
    实测（2026-10-05 E2E）：/api/parse 之后 session["images"]["bg"] 为空 →
      前端十图区全空、/api/regenerate 报「无背景图，无法重置」、
      /api/focus_drone 报 400「请先生成场景（无背景图）」。
    现在按「是否含 bg 槽位」自动判别，两种形态都能正确回填。
    """
    store = session if "bg" in session else session.setdefault(
        "images", _init_session_images(sid))
    for slot in _ALL_SLOTS:
        if slot in (gen_result.get("images") or {}):
            store[slot] = gen_result["images"][slot]


# ============================================================
# 页面路由
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# API 路由
# ============================================================

@app.route("/api/parse", methods=["POST"])
def api_parse():
    """
    一键生成：接收用户输入 → LLM 解析 → 搜索背景 → v7 全管线生成。

    请求:
        {
          "text": "...",                       # 必填
          "backend": "deepseek",               # deepseek|openai|claude|dry-run
          "skip_cnet": false,                  # true=不做 ControlNet 重绘（快速预览）
          "blend_steps": 15,                   # 重绘推理步数
          "blend_strength": 0.65,              # 重绘强度（有效步数 = int(步数×强度)）
          "cnet_scales": [0.85, 0.85],         # depth / seg 条件强度
          "use_lora": true
        }
    响应: {"ok": true, "background_spec": {...}, "full_scene_spec": {...},
           "session_id": 0, "images": {...}, "gen_params": {...}}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"ok": False, "error": "缺少 text 字段"}), 400

    user_text = data["text"].strip()
    backend = data.get("backend", "deepseek")
    knobs = _knobs(data, ("skip_cnet", "use_lora", "blend_steps",
                          "blend_strength", "cnet_scales"))

    try:
        result = parse_to_dual_json(user_text, backend=backend)

        sid = _alloc_sid()
        images = _init_session_images(sid)
        warnings = []

        # 搜索背景图
        bg_result = search_background(result["background_spec"], session_id=sid)
        if not bg_result["ok"]:
            return jsonify({"ok": False,
                            "error": "背景图搜索失败: " + bg_result.get("error", "未知")}), 500

        bg_url = bg_result["image_url"]
        bg_name = Path(bg_url).name
        bg_path = IMAGE_DIR / bg_name

        # 🔥 自动走 v7 全管线：LoRA 生成 → 抠图 → 合成 → 局部重绘
        try:
            from condition_generator_v7 import generate_with_lora
            print(f"[api/parse] sid={sid} knobs={knobs}", flush=True)
            gen_result = generate_with_lora(
                str(bg_path),
                result["full_scene_spec"],
                session_id=sid,
                **knobs,
            )
            if gen_result.get("ok"):
                _apply_generation(images, sid, gen_result)
            else:
                warnings.append("管线未返回 ok: " + str(gen_result.get("error", "未知")))
                images["bg"].append(bg_url)
        except Exception as e:
            # v7.2 修复：原来静默吞掉 ImportError，界面只显示一张背景图且无任何提示
            traceback.print_exc()
            warnings.append(f"生成管线异常（{type(e).__name__}: {e}）")
            images["bg"].append(bg_url)

        session_entry = {
            "id": sid,
            "timestamp": datetime.now().isoformat(),
            "input": user_text,
            "background_spec": result["background_spec"],
            "full_scene_spec": result["full_scene_spec"],
            "backend": backend,
            "gen_params": knobs,
            "images": images,
        }
        _put_session(session_entry)

        return jsonify({
            "ok": True,
            "background_spec": result["background_spec"],
            "full_scene_spec": result["full_scene_spec"],
            "session_id": sid,
            "images": images,
            "gen_params": knobs,
            "warnings": warnings,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/search-background/<int:sid>", methods=["POST"])
def api_search_background(sid):
    """手动重新搜索背景图（用于切换背景）。"""
    session = sessions.get(sid)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    bg_result = search_background(session["background_spec"], session_id=sid)

    if bg_result["ok"]:
        if "images" not in session:
            session["images"] = _init_session_images(sid)
        session["images"]["bg"] = [bg_result["image_url"]]
        return jsonify({"ok": True, "images": session["images"]})
    else:
        return jsonify({"ok": False, "error": bg_result.get("error", "搜索失败")}), 500


@app.route("/api/generate-lora/<int:sid>", methods=["POST"])
def api_generate_lora(sid):
    """
    v7 管线：LoRA 无人机生成 + rembg + crop-scale-blend。
    这是唯一默认开启 ControlNet 双条件重绘的端点（skip_cnet 默认 false）。

    请求体可选: skip_cnet / use_lora / blend_steps / blend_strength / cnet_scales
    """
    session = sessions.get(sid)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    bg_images = session.get("images", {}).get("bg", [])
    if not bg_images:
        return jsonify({"ok": False, "error": "请先搜索背景图"}), 400

    bg_url = bg_images[0]
    bg_name = Path(bg_url).name
    bg_path = IMAGE_DIR / bg_name
    if not bg_path.exists():
        return jsonify({"ok": False, "error": f"背景图文件不存在: {bg_name}"}), 404

    knobs = _knobs(request.get_json(silent=True),
                   ("skip_cnet", "use_lora", "blend_steps",
                    "blend_strength", "cnet_scales"))

    try:
        from condition_generator_v7 import generate_with_lora

        print(f"[api/generate-lora] sid={sid} knobs={knobs}", flush=True)
        result = generate_with_lora(
            str(bg_path),
            session.get("full_scene_spec", {}),
            session_id=sid,
            **knobs,
        )

        if result.get("ok"):
            _apply_generation(session, sid, result)
            return jsonify({
                "ok": True,
                "images": session["images"],
                "frame_count": result.get("frame_count", 1),
                "trajectory_summary": result.get("trajectory_summary", []),
                "gen_params": knobs,
            })
        else:
            return jsonify({"ok": False, "error": result.get("error", "生成失败")}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regenerate/<int:sid>", methods=["POST"])
def api_regenerate(sid):
    """
    重置指定 slot 及所有下游 slot。

    依赖链:
      bg → bg_depth, bg_seg → drone_depth, drone_seg → final_rgb
      drone_raw → drone_clean → drone_depth, drone_seg → final_rgb

    ⚠️ 实现上目前是整条管线重跑一遍后回填，而非只重算下游子集；
       `regenerated` 字段返回的才是语义上的受影响集合。

    请求: {"slot": "bg", "skip_cnet": false, "blend_steps": 15, ...}
    """
    session = sessions.get(sid)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    data = request.get_json(silent=True) or {}
    slot = data.get("slot", "")
    knobs = _knobs(data, ("skip_cnet", "use_lora", "blend_steps",
                          "blend_strength", "cnet_scales"))

    # 如果重置 bg，先重新搜索背景
    if slot == "bg":
        bg_result = search_background(session["background_spec"], session_id=sid)
        if not bg_result["ok"]:
            return jsonify({"ok": False,
                            "error": "背景搜索失败: " + bg_result.get("error", "未知")}), 500
        bg_url = bg_result["image_url"]
    else:
        bg_images = session.get("images", {}).get("bg", [])
        if not bg_images:
            return jsonify({"ok": False, "error": "无背景图，无法重置"}), 400
        bg_url = bg_images[0]

    bg_name = Path(bg_url).name
    bg_path = IMAGE_DIR / bg_name
    if not bg_path.exists():
        return jsonify({"ok": False, "error": f"背景图文件不存在: {bg_name}"}), 404

    # 下游依赖映射
    downstream_map = {
        "bg":          ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "bg_depth":    ["bg_depth", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "bg_seg":      ["bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_raw":   ["drone_raw", "drone_clean", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_clean": ["drone_clean", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_depth": ["drone_depth", "final_rgb", "ir", "bbox"],
        "drone_seg":   ["drone_seg", "final_rgb", "ir", "bbox"],
        "final_rgb":   ["final_rgb", "ir", "bbox"],
        "ir":          ["ir"],
        "bbox":        ["bbox"],
    }
    if slot not in downstream_map:
        return jsonify({"ok": False, "error": f"未知 slot: {slot!r}"}), 400

    try:
        from condition_generator_v7 import generate_with_lora

        print(f"[api/regenerate] sid={sid} slot={slot} knobs={knobs}", flush=True)
        gen_result = generate_with_lora(
            str(bg_path),
            session["full_scene_spec"],
            session_id=sid,
            **knobs,
        )

        if gen_result.get("ok"):
            _apply_generation(session, sid, gen_result)
            return jsonify({
                "ok": True,
                "images": session["images"],
                "slot": slot,
                "regenerated": downstream_map.get(slot, [slot]),
                "gen_params": knobs,
            })
        else:
            return jsonify({"ok": False, "error": gen_result.get("error", "生成失败")}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/focus_drone/<int:sid>", methods=["POST"])
def api_focus_drone(sid):
    """
    快速重定位无人机：前端改变红点标记后，仅重新合成 + 注入 + 重绘下游 slot。
    不重新生成 LoRA 无人机、不重新做 rembg、不重新分析背景 depth/seg。

    请求: {"dronePos": {"x": 50, "y": 30}, "droneScale": 0.15,
           "skip_cnet": false, "blend_steps": 15, ...}
    响应: {"ok": true, "images": {...}, "nu": 0.5, "nv": 0.3, "target_px": ..., "dval": ...}
    """
    session = sessions.get(sid)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    data = request.get_json(silent=True) or {}
    drone_pos = data.get("dronePos") or {"x": 50, "y": 30}
    drone_scale = float(data.get("droneScale", 0.15))
    knobs = _knobs(data, ("skip_cnet", "use_lora", "blend_steps",
                          "blend_strength", "cnet_scales"))

    bg_images = session.get("images", {}).get("bg", [])
    if not bg_images:
        return jsonify({"ok": False, "error": "请先生成场景（无背景图）"}), 400

    bg_name = Path(bg_images[0]).name
    bg_path = IMAGE_DIR / bg_name
    if not bg_path.exists():
        return jsonify({"ok": False, "error": f"背景图不存在: {bg_name}"}), 404

    drone_rgba_files = sorted(IMAGE_DIR.glob(f"session_{sid}_drone_rgba_*.png"))
    if not drone_rgba_files:
        return jsonify({"ok": False, "error": "请先生成无人机（无 drone_rgba）"}), 400

    drone_rgba_path = str(drone_rgba_files[0])  # 取第一帧的 RGBA

    try:
        from condition_generator_v7 import focus_reposition

        print(f"[api/focus_drone] sid={sid} pos=({drone_pos.get('x')},{drone_pos.get('y')}) "
              f"scale={drone_scale} knobs={knobs}", flush=True)
        result = focus_reposition(
            str(bg_path),
            drone_rgba_path,
            session.get("full_scene_spec", {}),
            drone_pos_x=float(drone_pos.get("x", 50)),
            drone_pos_y=float(drone_pos.get("y", 30)),
            drone_scale=drone_scale,
            session_id=sid,
            **knobs,
        )

        if result.get("ok"):
            if "images" not in session:
                session["images"] = _init_session_images(sid)
            # 只更新下游 slot（不碰 bg/bg_depth/bg_seg/drone_raw/drone_clean）
            for slot in ("drone_depth", "drone_seg", "final_rgb", "ir", "bbox"):
                if slot in result["images"]:
                    session["images"][slot] = result["images"][slot]

            return jsonify({
                "ok": True,
                "images": session["images"],
                "nu": result["nu"],
                "nv": result["nv"],
                "target_px": result["target_px"],
                "dval": result["dval"],
                "gen_params": knobs,
            })
        else:
            return jsonify({"ok": False, "error": result.get("error", "重定位失败")}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    """返回所有会话记录（按 id 升序）。"""
    return jsonify({"sessions": _session_list()})


@app.route("/api/session/<int:sid>", methods=["GET"])
def api_session(sid):
    """返回单条会话记录。"""
    session = sessions.get(sid)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"session": session})


@app.route("/api/session/<int:sid>/images", methods=["POST"])
def api_upload_images(sid):
    """
    上传生成图片（供后续生成端 Agent 调用）。
    请求: multipart form, 字段名对应 slot 名
      bg, bg_depth, bg_seg, drone_depth, drone_seg, final_rgb, ir, bbox,
      drone_clean, drone_raw
    每个字段可传多张图（多帧）。
    """
    if sid not in sessions:
        return jsonify({"error": "session not found"}), 404

    if "images" not in sessions[sid]:
        sessions[sid]["images"] = _init_session_images(sid)

    img_store = sessions[sid]["images"]
    saved = {}

    for slot in _ALL_SLOTS:
        files = request.files.getlist(slot)
        if files:
            slot_urls = []
            for idx, f in enumerate(files):
                ext = Path(f.filename).suffix if f.filename else ".png"
                save_name = f"session_{sid}_{slot}_{idx}{ext}"
                save_path = IMAGE_DIR / save_name
                f.save(str(save_path))
                slot_urls.append(f"/images/{save_name}")
            img_store[slot] = slot_urls
            saved[slot] = slot_urls

    return jsonify({"ok": True, "saved": saved})


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(str(IMAGE_DIR), filename)


@app.route("/api/health", methods=["GET"])
def api_health():
    """轻量健康检查：便于脚本判断服务是否已就绪。"""
    with _LOCK:
        n = len(sessions)
    return jsonify({"ok": True, "sessions": n, "next_sid": _SID_SEQ})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """关闭 Flask 服务器。"""
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"ok": True, "message": "shutting down"})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════╗
║  低慢小数据集 — LLM Parser Web 前端 v7.2             ║
╠══════════════════════════════════════════════════════╣
║  地址: http://{host}:{port:<35}║
║  模式: 双 JSON + 10图展示 + 背景搜索                 ║
║  后端: DeepSeek (默认)                               ║
║  按 Ctrl+C 停止                                      ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(host=host, port=port, debug=debug)
