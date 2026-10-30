#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web 应用端点冒烟测试（v7.5, 2026-10-11）
=========================================
不加载任何模型、不占 GPU：用 Flask test_client 打通全部 12 个路由，
把 parse_to_dual_json / search_background / generate_with_lora /
focus_reposition 全部替换为桩函数，只验证「路由可达 + JSON 契约 + 旋钮透传」。

之所以写这个脚本：`/api/search-background`、`/api/generate-lora`、
`/api/session/<sid>/images` 三个端点前端从不调用（legacy），长期无人验证。
本轮把三者纳入自动化冒烟，避免它们悄悄腐烂。

用法（在 0-workspace/app 下）：
    python3 smoke_endpoints.py                 # 结果打印到 stdout
    python3 smoke_endpoints.py --json out.json # 同时落盘 JSON 报告
退出码：0 = 全部通过，1 = 有失败项。
"""

import argparse
import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── 桩函数必须在 import web_app 之前准备好（web_app 顶层已 import 部分符号）──
import web_app  # noqa: E402
import condition_generator_v7 as cg7  # noqa: E402

RESULTS = []          # [(name, ok, detail)]
CALLS = []            # 桩调用记录


def _record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  ← {detail}"))
    return ok


# ══════════════════════════════════════════════════════════════════
# 桩
# ══════════════════════════════════════════════════════════════════
_FAKE_BG = "bg_fixture.jpg"


def stub_search_background(bg_spec, session_id=0):
    CALLS.append(("search_background", {"session_id": session_id}))
    return {
        "ok": True,
        "image_url": f"/images/{_FAKE_BG}",
        "source_path": str(web_app.IMAGE_DIR / _FAKE_BG),
        "matched_labels": ("noon", "clear"),
        "match_level": "tod+weather",
        "candidates": 42,
    }


_SLOT_URL = "/images/stub.png"


def stub_parse_to_dual_json(user_input, backend="dry", **kw):
    CALLS.append(("parse_to_dual_json", {"backend": backend}))
    return {
        "background_spec": {"scene_type": "urban", "time_of_day": "noon",
                            "weather": "clear"},
        "full_scene_spec": {"scene_type": "urban", "drones": []},
    }


def _gen_result():
    imgs = {s: [_SLOT_URL] for s in web_app._ALL_SLOTS}
    return {"ok": True, "images": imgs, "frame_count": 1}


def stub_generate_with_lora(bg_path, full_scene_spec, session_id=0, **knobs):
    CALLS.append(("generate_with_lora", {"session_id": session_id, **knobs}))
    return _gen_result()


def stub_focus_reposition(bg_path, drone_rgba, spec, session_id=0, **kw):
    CALLS.append(("focus_reposition", {"session_id": session_id, **kw}))
    r = _gen_result()
    r.update({"nu": 0.5, "nv": 0.3, "target_px": 96, "dval": 120})
    return r


def install_stubs(img_dir: Path):
    web_app.parse_to_dual_json = stub_parse_to_dual_json
    web_app.search_background = stub_search_background
    cg7.generate_with_lora = stub_generate_with_lora
    cg7.focus_reposition = stub_focus_reposition
    web_app.IMAGE_DIR = img_dir


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="把报告写到该 JSON 路径")
    args = ap.parse_args()

    web_app.app.config["TESTING"] = True
    client = web_app.app.test_client()

    with tempfile.TemporaryDirectory() as td:
        img_dir = Path(td)
        # 背景与 drone_rgba 夹具：端点只做 exists() 检查，内容无关
        (img_dir / _FAKE_BG).write_bytes(b"\xff\xd8\xff\xd9")
        (img_dir / "stub.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # _SLOT_URL 目标
        install_stubs(img_dir)

        print("\n【A. 静态页面】")
        r = client.get("/")
        _record("GET / → 200 text/html", r.status_code == 200
                and "text/html" in r.headers.get("Content-Type", ""),
                f"{r.status_code}")
        _record("GET / 含关键 UI（cnet_scales/use_lora 控件已暴露）",
                b"opt-cnet-depth" in r.data and b"opt-use-lora" in r.data
                and b"opt-cnet-seg" in r.data)

        print("\n【B. /api/parse（一键链路）】")
        r = client.post("/api/parse", json={})
        _record("POST /api/parse 缺 text → 400", r.status_code == 400,
                f"{r.status_code}")
        r = client.post("/api/parse", json={"text": "城市仰拍", "backend": "dry"})
        d = r.get_json()
        _record("POST /api/parse → 200 ok", r.status_code == 200 and d.get("ok"),
                f"{r.status_code} {d}")
        sid = d.get("session_id")
        _record("→ 返回 session_id(int)", isinstance(sid, int), repr(sid))
        _record("→ 返回 10 槽位 images",
                isinstance(d.get("images"), dict)
                and len(d.get("images", {})) == len(web_app._ALL_SLOTS),
                str(sorted((d.get("images") or {}).keys())))
        _record("→ 回填非空（_apply_generation 形态判别修复）",
                bool((d.get("images") or {}).get("final_rgb")),
                str(d.get("images", {}).get("final_rgb")))

        print("\n【C. 旋钮透传（前端 pipelineKnobs → 后端 _knobs）】")
        payload = {"text": "沙漠飞近", "backend": "dry", "skip_cnet": False,
                   "use_lora": False, "blend_steps": 12, "blend_strength": 0.55,
                   "cnet_scales": [0.30, 0.90]}
        r = client.post("/api/parse", json=payload)
        d = r.get_json()
        gp = d.get("gen_params") or {}
        _record("parse: use_lora=False 透传", gp.get("use_lora") is False, repr(gp))
        # 注：后端内部是元组（见 [api/parse] knobs=... 日志），jsonify 会把元组序列化成列表
        _record("parse: cnet_scales=(0.30,0.90) 透传",
                tuple(gp.get("cnet_scales") or ()) == (0.30, 0.90),
                repr(gp.get("cnet_scales")))
        _record("parse: blend_steps/blend_strength 透传",
                gp.get("blend_steps") == 12 and abs(gp.get("blend_strength", 0) - 0.55) < 1e-9,
                repr(gp))
        calls = [c for c in CALLS if c[0] == "generate_with_lora"]
        last = calls[-1][1] if calls else {}
        _record("parse: 旋钮真正传进 generate_with_lora",
                last.get("use_lora") is False
                and last.get("cnet_scales") == (0.30, 0.90)
                and last.get("blend_steps") == 12, repr(last))

        print("\n【D. legacy 端点①  /api/search-background/<sid>】")
        r = client.post(f"/api/search-background/{sid}", json={})
        d = r.get_json()
        _record("POST /api/search-background/<sid> → 200 ok",
                r.status_code == 200 and d.get("ok"), f"{r.status_code} {d}")
        _record("→ images.bg 被替换", bool((d.get("images") or {}).get("bg")),
                str((d.get("images") or {}).get("bg")))
        r = client.post("/api/search-background/99999", json={})
        _record("→ 未知 sid → 404", r.status_code == 404, f"{r.status_code}")

        print("\n【E. legacy 端点②  /api/generate-lora/<sid>】")
        r = client.post(f"/api/generate-lora/{sid}",
                        json={"skip_cnet": True, "use_lora": True,
                              "cnet_scales": [0.85, 0.85]})
        d = r.get_json()
        _record("POST /api/generate-lora/<sid> → 200 ok",
                r.status_code == 200 and d.get("ok"), f"{r.status_code} {d}")
        _record("→ 返回 gen_params（skip_cnet=True / cnet_scales）",
                (d.get("gen_params") or {}).get("skip_cnet") is True
                and tuple((d.get("gen_params") or {}).get("cnet_scales") or ()) == (0.85, 0.85),
                str(d.get("gen_params")))
        r = client.post("/api/generate-lora/99999", json={})
        _record("→ 未知 sid → 404", r.status_code == 404, f"{r.status_code}")

        # 无背景图会话 → 400
        sid_nobg = web_app._alloc_sid()
        web_app._put_session({"id": sid_nobg, "background_spec": {},
                              "full_scene_spec": {},
                              "images": web_app._init_session_images(sid_nobg)})
        r = client.post(f"/api/generate-lora/{sid_nobg}", json={})
        _record("→ 无背景图 → 400", r.status_code == 400, f"{r.status_code}")

        print("\n【F. legacy 端点③  /api/session/<sid>/images（上传）】")
        # Focus_drone 之前先做上传（BytesIO 一次性，故每次重建 payload）
        def _upd():
            return {"final_rgb": (io.BytesIO(b"fake-png"), "a.png"),
                    "ir": (io.BytesIO(b"fake-png"), "b.png")}

        data = _upd()
        r = client.post(f"/api/session/{sid}/images", data=data,
                        content_type="multipart/form-data")
        d = r.get_json()
        _record("POST /api/session/<sid>/images → 200 ok",
                r.status_code == 200 and d.get("ok"), f"{r.status_code} {d}")
        _record("→ saved 含两个槽位且 URL 形如 /images/session_<sid>_<slot>_0.png",
                set((d.get("saved") or {}).keys()) == {"final_rgb", "ir"}
                and all(u.startswith(f"/images/session_{sid}_")
                        for u in (d.get("saved") or {}).get("final_rgb", [])),
                str(d.get("saved")))
        r = client.post("/api/session/99999/images", data=_upd(),
                        content_type="multipart/form-data")
        _record("→ 未知 sid → 404", r.status_code == 404, f"{r.status_code}")

        print("\n【G. 前端在用的其余端点】")
        r = client.post(f"/api/regenerate/{sid}", json={"slot": "final_rgb"})
        d = r.get_json()
        _record("POST /api/regenerate/<sid> slot=final_rgb → 200",
                r.status_code == 200 and d.get("ok"), f"{r.status_code} {d}")
        _record("→ regenerated 依赖链正确",
                (d.get("regenerated") or []) == ["final_rgb", "ir", "bbox"],
                str(d.get("regenerated")))
        r = client.post(f"/api/regenerate/{sid}", json={"slot": "bogus"})
        _record("→ 未知 slot → 400", r.status_code == 400, f"{r.status_code}")
        r = client.post("/api/regenerate/99999", json={"slot": "bg"})
        _record("→ 未知 sid → 404", r.status_code == 404, f"{r.status_code}")

        # focus_drone 需要 drone_rgba 夹具
        (img_dir / f"session_{sid}_drone_rgba_0.png").write_bytes(b"fake-png")
        r = client.post(f"/api/focus_drone/{sid}",
                        json={"dronePos": {"x": 60, "y": 40}, "droneScale": 0.12,
                              "cnet_scales": [0.85, 0.85], "use_lora": True})
        d = r.get_json()
        _record("POST /api/focus_drone/<sid> → 200 ok",
                r.status_code == 200 and d.get("ok"), f"{r.status_code} {d}")
        _record("→ 返回 nu/nv/target_px/dval",
                all(k in d for k in ("nu", "nv", "target_px", "dval")), str(d))

        r = client.get("/api/sessions")
        d = r.get_json()
        _record("GET /api/sessions → 200 且含 sessions 列表",
                r.status_code == 200 and isinstance(d.get("sessions"), list),
                f"{r.status_code} n={len((d.get('sessions') or []))}")
        r = client.get(f"/api/session/{sid}")
        _record("GET /api/session/<sid> → 200 含 session",
                r.status_code == 200 and isinstance(r.get_json().get("session"), dict),
                f"{r.status_code}")
        r = client.get("/api/session/99999")
        _record("→ 未知 sid → 404", r.status_code == 404, f"{r.status_code}")

        r = client.get(f"/images/{_FAKE_BG}")
        _record("GET /images/<name> → 200", r.status_code == 200, f"{r.status_code}")
        r = client.get("/api/health")
        d = r.get_json()
        _record("GET /api/health → 200 ok + sessions/next_sid",
                r.status_code == 200 and d.get("ok")
                and "sessions" in d and "next_sid" in d, str(d))

        print("\n【H. /api/shutdown（拦截 SIGTERM，只验路由）】")
        _real_kill = os.kill
        os.kill = lambda *a, **k: None
        try:
            r = client.post("/api/shutdown")
            _record("POST /api/shutdown → 200 ok",
                    r.status_code == 200 and r.get_json().get("ok"),
                    f"{r.status_code}")
        finally:
            os.kill = _real_kill

    # ── 汇总 ──────────────────────────────────────────────────
    total = len(RESULTS)
    bad = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 66)
    print(f"冒烟结果：{total - len(bad)}/{total} 通过")
    if bad:
        print("失败项：")
        for n, _, det in bad:
            print(f"  ✗ {n}  ← {det}")
    print("=" * 66)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"total": total, "passed": total - len(bad),
             "failed": [{"name": n, "detail": d} for n, _, d in bad],
             "results": [{"name": n, "ok": o, "detail": d} for n, o, d in RESULTS]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写：{args.json}")

    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
