"""
低慢小数据集生成架构 — LLM Parser Web 服务 (v6.0)
=================================================
本地 Web 前端 + API，双 JSON 输出模式：
  - background_spec.json — Agent 2 背景匹配
  - full_scene_spec.json  — Agent 3/4/5/6 全管线

启动: python web_app.py
访问: http://127.0.0.1:5000
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

from llm_parser import parse, parse_to_dual_json
from background_searcher import search_background, get_placeholder_image

# 加载 .env
load_dotenv(Path(__file__).parent / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# 会话存储（简易内存存储，重启丢失）
sessions: list[dict] = []

# 图片输出目录
IMAGE_DIR = Path(__file__).parent / "output_images"
IMAGE_DIR.mkdir(exist_ok=True)


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
    请求: {"text": "...", "backend": "deepseek"}
    响应: {"ok": true, "background_spec": {...}, "full_scene_spec": {...}, "session_id": 0, "images": {...}}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"ok": False, "error": "缺少 text 字段"}), 400

    user_text = data["text"].strip()
    backend = data.get("backend", "deepseek")

    try:
        result = parse_to_dual_json(user_text, backend=backend)

        sid = len(sessions)

        # 初始化图片存储（全空，不生成占位图）
        images = _init_session_images(sid)

        # 搜索背景图
        bg_result = search_background(result["background_spec"], session_id=sid)
        if not bg_result["ok"]:
            return jsonify({"ok": False, "error": "背景图搜索失败: " + bg_result.get("error", "未知")}), 500

        bg_url = bg_result["image_url"]
        bg_name = Path(bg_url).name
        bg_path = IMAGE_DIR / bg_name

        # 🔥 自动走 v7 全管线：LoRA 生成 → 抠图 → 合成 → 局部重绘
        try:
            from condition_generator_v7 import generate_with_lora
            gen_result = generate_with_lora(
                str(bg_path),
                result["full_scene_spec"],
                session_id=sid,
                skip_cnet=True,  # 暂时跳过 ControlNet 重绘
            )
            if gen_result["ok"]:
                for slot in ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox", "drone_clean", "drone_raw"]:
                    if slot in gen_result["images"]:
                        images[slot] = gen_result["images"][slot]
            else:
                # 生成失败，至少保留背景图
                images["bg"].append(bg_url)
        except ImportError:
            images["bg"].append(bg_url)

        # 记录会话
        session_entry = {
            "id": sid,
            "timestamp": datetime.now().isoformat(),
            "input": user_text,
            "background_spec": result["background_spec"],
            "full_scene_spec": result["full_scene_spec"],
            "backend": backend,
            "images": images,
        }
        sessions.append(session_entry)

        return jsonify({
            "ok": True,
            "background_spec": result["background_spec"],
            "full_scene_spec": result["full_scene_spec"],
            "session_id": sid,
            "images": images,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/search-background/<int:sid>", methods=["POST"])
def api_search_background(sid):
    """
    手动重新搜索背景图（用于切换背景）。
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    session = sessions[sid]
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
    v7 管线：LoRA 无人机生成 + rembg + crop-scale-blend（替代 Gaussian blob）。
    需先有背景图。产物含 drone_clean（LoRA 抠图后）。
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    session = sessions[sid]
    bg_images = session.get("images", {}).get("bg", [])
    if not bg_images:
        return jsonify({"ok": False, "error": "请先搜索背景图"}), 400

    bg_url = bg_images[0]
    bg_name = Path(bg_url).name
    bg_path = IMAGE_DIR / bg_name
    if not bg_path.exists():
        return jsonify({"ok": False, "error": f"背景图文件不存在: {bg_name}"}), 404

    try:
        from condition_generator_v7 import generate_with_lora

        result = generate_with_lora(
            str(bg_path),
            session.get("full_scene_spec", {}),
            session_id=sid,
        )

        if result["ok"]:
            if "images" not in session:
                session["images"] = _init_session_images(sid)

            for slot in ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox", "drone_clean", "drone_raw"]:
                if slot in result["images"]:
                    session["images"][slot] = result["images"][slot]

            return jsonify({
                "ok": True,
                "images": session["images"],
                "frame_count": result.get("frame_count", 1),
                "trajectory_summary": result.get("trajectory_summary", []),
            })
        else:
            return jsonify({"ok": False, "error": result.get("error", "生成失败")}), 500

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/regenerate/<int:sid>", methods=["POST"])
def api_regenerate(sid):
    """
    重置指定 slot 及所有下游 slot。
    依赖链:
      bg → bg_depth, bg_seg → drone_depth, drone_seg → final_rgb
      drone_raw → drone_clean → drone_depth, drone_seg → final_rgb
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    data = request.get_json()
    slot = data.get("slot", "")

    session = sessions[sid]

    # 如果重置 bg，先重新搜索背景
    if slot == "bg":
        bg_result = search_background(session["background_spec"], session_id=sid)
        if not bg_result["ok"]:
            return jsonify({"ok": False, "error": "背景搜索失败: " + bg_result.get("error", "未知")}), 500
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
        "bg":         ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "bg_depth":   ["bg_depth", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "bg_seg":     ["bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_raw":  ["drone_raw", "drone_clean", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_clean":["drone_clean", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox"],
        "drone_depth":["drone_depth", "final_rgb", "ir", "bbox"],
        "drone_seg":  ["drone_seg", "final_rgb", "ir", "bbox"],
        "final_rgb":  ["final_rgb", "ir", "bbox"],
        "ir":         ["ir"],
        "bbox":       ["bbox"],
    }

    try:
        from condition_generator_v7 import generate_with_lora

        gen_result = generate_with_lora(
            str(bg_path),
            session["full_scene_spec"],
            session_id=sid,
            skip_cnet=True,
        )

        if gen_result["ok"]:
            if "images" not in session:
                session["images"] = _init_session_images(sid)

            for s in ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg",
                       "final_rgb", "ir", "bbox", "drone_clean", "drone_raw"]:
                if s in gen_result["images"]:
                    session["images"][s] = gen_result["images"][s]

            return jsonify({
                "ok": True,
                "images": session["images"],
                "slot": slot,
                "regenerated": downstream_map.get(slot, [slot]),
            })
        else:
            return jsonify({"ok": False, "error": gen_result.get("error", "生成失败")}), 500

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/focus_drone/<int:sid>", methods=["POST"])
def api_focus_drone(sid):
    """
    快速重定位无人机：前端改变红点标记后，仅重新合成+注入+重绘下游 slot。
    不重新生成 LoRA 无人机、不重新做 rembg、不重新分析背景 depth/seg。

    请求: {"dronePos": {"x": 50, "y": 30}, "droneScale": 0.15}
    响应: {"ok": true, "images": {...}, "nu": 0.5, "nv": 0.3}
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    session = sessions[sid]
    data = request.get_json() or {}
    drone_pos = data.get("dronePos", {"x": 50, "y": 30})
    drone_scale = float(data.get("droneScale", 0.15))

    # 必须有背景图
    bg_images = session.get("images", {}).get("bg", [])
    if not bg_images:
        return jsonify({"ok": False, "error": "请先生成场景（无背景图）"}), 400

    bg_name = Path(bg_images[0]).name
    bg_path = IMAGE_DIR / bg_name
    if not bg_path.exists():
        return jsonify({"ok": False, "error": f"背景图不存在: {bg_name}"}), 404

    # 必须有 drone_rgba
    drone_rgba_files = list(IMAGE_DIR.glob(f"session_{sid}_drone_rgba_*.png"))
    if not drone_rgba_files:
        return jsonify({"ok": False, "error": "请先生成无人机（无 drone_rgba）"}), 400

    drone_rgba_path = str(drone_rgba_files[0])  # 取第一帧的 RGBA

    try:
        from condition_generator_v7 import focus_reposition

        result = focus_reposition(
            str(bg_path),
            drone_rgba_path,
            session.get("full_scene_spec", {}),
            drone_pos_x=float(drone_pos.get("x", 50)),
            drone_pos_y=float(drone_pos.get("y", 30)),
            drone_scale=drone_scale,
            session_id=sid,
            skip_cnet=True,  # 暂时跳过 ControlNet 重绘（速度优先）
        )

        if result["ok"]:
            if "images" not in session:
                session["images"] = _init_session_images(sid)

            # 只更新下游 slot（不碰 bg/bg_depth/bg_seg/drone_raw/drone_clean）
            for slot in ["drone_depth", "drone_seg", "final_rgb", "ir", "bbox"]:
                if slot in result["images"]:
                    session["images"][slot] = result["images"][slot]

            return jsonify({
                "ok": True,
                "images": session["images"],
                "nu": result["nu"],
                "nv": result["nv"],
                "target_px": result["target_px"],
                "dval": result["dval"],
            })
        else:
            return jsonify({"ok": False, "error": result.get("error", "重定位失败")}), 500

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    """返回所有会话记录。"""
    return jsonify({"sessions": sessions})


@app.route("/api/session/<int:sid>", methods=["GET"])
def api_session(sid):
    """返回单条会话记录。"""
    if 0 <= sid < len(sessions):
        return jsonify({"session": sessions[sid]})
    return jsonify({"error": "session not found"}), 404


@app.route("/api/session/<int:sid>/images", methods=["POST"])
def api_upload_images(sid):
    """
    上传生成图片（供后续生成端 Agent 调用）。
    请求: multipart form, 字段名对应 slot 名
      bg, bg_depth, bg_seg, drone_depth, drone_seg, final_rgb
    每个字段可传多张图（多帧）。
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    if "images" not in sessions[sid]:
        sessions[sid]["images"] = _init_session_images(sid)

    img_store = sessions[sid]["images"]
    saved = {}

    for slot in ["bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg", "final_rgb", "ir", "bbox", "drone_clean"]:
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
╔══════════════════════════════════════════════╗
║  低慢小数据集 — LLM Parser Web 前端 v7.0    ║
╠══════════════════════════════════════════════╣
║  地址: http://{host}:{port}                  ║
║  模式: 双 JSON + 6图展示 + 背景搜索          ║
║  后端: DeepSeek (默认)                       ║
║  按 Ctrl+C 停止                              ║
╚══════════════════════════════════════════════╝
""")
    app.run(host=host, port=port, debug=debug)
