"""
低慢小数据集生成架构 — LLM Parser Web 服务
===========================================
本地 Web 前端 + API，包含：
  - 会话记录窗口
  - LLM 对话输入
  - RGB / IR 生成图展示

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

from llm_parser import parse, scene_spec_to_dict

# 加载 .env
load_dotenv(Path(__file__).parent / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# 会话存储（简易内存存储，重启丢失）
sessions: list[dict] = []

# 图片输出目录
IMAGE_DIR = Path(__file__).parent / "output_images"
IMAGE_DIR.mkdir(exist_ok=True)


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
    接收用户输入，调用 LLM 解析为 JSON。
    请求: {"text": "...", "backend": "deepseek"}
    响应: {"ok": true, "spec": {...}, "session_id": 0}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"ok": False, "error": "缺少 text 字段"}), 400

    user_text = data["text"].strip()
    backend = data.get("backend", "deepseek")

    try:
        spec = parse(user_text, backend=backend)
        spec_dict = scene_spec_to_dict(spec)

        # 记录会话
        session_entry = {
            "id": len(sessions),
            "timestamp": datetime.now().isoformat(),
            "input": user_text,
            "spec": spec_dict,
            "backend": backend,
            "rgb_image": None,  # 后续生成端填充
            "ir_image": None,
        }
        sessions.append(session_entry)

        return jsonify({
            "ok": True,
            "spec": spec_dict,
            "session_id": session_entry["id"],
        })

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
    请求: multipart form, rgb=file, ir=file
    """
    if not (0 <= sid < len(sessions)):
        return jsonify({"error": "session not found"}), 404

    rgb_file = request.files.get("rgb")
    ir_file = request.files.get("ir")

    saved = {}
    if rgb_file:
        rgb_path = IMAGE_DIR / f"session_{sid}_rgb.png"
        rgb_file.save(str(rgb_path))
        sessions[sid]["rgb_image"] = f"/images/session_{sid}_rgb.png"
        saved["rgb"] = sessions[sid]["rgb_image"]

    if ir_file:
        ir_path = IMAGE_DIR / f"session_{sid}_ir.png"
        ir_file.save(str(ir_path))
        sessions[sid]["ir_image"] = f"/images/session_{sid}_ir.png"
        saved["ir"] = sessions[sid]["ir_image"]

    return jsonify({"ok": True, "saved": saved})


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(str(IMAGE_DIR), filename)


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════╗
║  低慢小数据集 — LLM Parser Web 前端          ║
╠══════════════════════════════════════════════╣
║  地址: http://{host}:{port}                  ║
║  后端: DeepSeek (默认)                        ║
║  按 Ctrl+C 停止                               ║
╚══════════════════════════════════════════════╝
""")
    app.run(host=host, port=port, debug=debug)
