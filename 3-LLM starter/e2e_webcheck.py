#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_app.py 端到端验证（GPU 必需，一次性跑通）

覆盖：
  1. /api/health 就绪探测
  2. /api/parse  dry 后端全管线（LoRA 生成 → 贴入 → CN 重绘）
  3. session images 十槽位完整性 + 每个 URL 可回取（HTTP 200）
  4. /api/regenerate 并发 2 请求（验证 _PIPE_LOCK / _LOCK 串行化，不出 CUDA 错）
  5. /api/focus_drone
  6. /api/session/<sid>/images 保存 + /api/session(s) 读回
  7. /api/shutdown 优雅退出
  8. 结束后检查 stdout 无 Traceback / RuntimeError / CUDA error

用法：python3 -u e2e_webcheck.py [--port 5077] [--timeout 900]
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request, urllib.error, signal

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = None
RESULTS = []


def rec(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def http(path, payload=None, method=None, timeout=600):
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    m = method or ("POST" if data is not None else "GET")
    req = urllib.request.Request(url, data=data, method=m,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def http_raw(path, timeout=300):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), len(r.read())
    except Exception as e:
        return 0, str(e), 0


def http_multipart(path, files):
    """以 multipart/form-data 上传假 PNG（测试 /api/session/<sid>/images）。"""
    import uuid
    b = "----e2e" + uuid.uuid4().hex
    png = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                        "0000000a49444154789c6300010000050001")
    buf = b""
    for slot, fn in files.items():
        buf += (f"--{b}\r\nContent-Disposition: form-data; name=\"{slot}\"; "
                f"filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        buf += png + b"\r\n"
    buf += f"--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=buf, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def collect_urls(images):
    urls = []
    for slot, v in (images or {}).items():
        if isinstance(v, str):
            urls.append((slot, v))
        elif isinstance(v, list):
            for i, u in enumerate(v):
                if isinstance(u, str):
                    urls.append((f"{slot}[{i}]", u))
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5077)
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    global BASE
    BASE = f"http://127.0.0.1:{a.port}"

    log = os.path.join(ROOT, "_e2e_stdout.log")
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, FLASK_PORT=str(a.port), FLASK_HOST="127.0.0.1", FLASK_DEBUG="false")
    print(f"===== web_app E2E  {time.strftime('%F %T')}  port={a.port} =====", flush=True)
    lf = open(log, "wb")
    proc = subprocess.Popen([sys.executable, "-u", "web_app.py"], cwd=ROOT, env=env,
                            stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)

    t0 = time.time()
    ready = False
    while time.time() - t0 < a.timeout:
        if proc.poll() is not None:
            rec("server alive", False, f"进程提前退出 code={proc.returncode}")
            break
        try:
            st, j = http("/api/health", timeout=5)
            if st == 200 and isinstance(j, dict) and j.get("ok"):
                ready = True
                rec("1. /api/health 就绪", True, f"{time.time()-t0:.0f}s  {j}")
                break
        except Exception:
            pass
        time.sleep(2)
    if not ready and not RESULTS:
        rec("1. /api/health 就绪", False, f"{a.timeout}s 超时")

    sid = None
    need = {"bg", "bg_depth", "bg_seg", "drone_depth", "drone_seg",
            "final_rgb", "ir", "bbox", "drone_clean", "drone_raw"}
    if ready:
        # ── 2. /api/parse 全管线 ──
        t = time.time()
        st, j = http("/api/parse", {"text": "一架小型四旋翼无人机在清晨城市上空悬停",
                                    "backend": "dry", "skip_cnet": False,
                                    "blend_steps": 15, "blend_strength": 0.65,
                                    "cnet_scales": [0.85, 0.85], "use_lora": True})
        ok = st == 200 and isinstance(j, dict) and j.get("ok")
        rec("2. /api/parse 全管线(skip_cnet=false)", ok, f"HTTP {st} {time.time()-t:.1f}s")
        if ok:
            sid = j.get("session_id")
            rec("   - 返回 session_id", isinstance(sid, int), f"sid={sid}")
            rec("   - 两个 JSON 都返回", bool(j.get("background_spec")) and bool(j.get("full_scene_spec")),
                f"bg_keys={list((j.get('background_spec') or {}).keys())[:4]}")
            gp = j.get("gen_params") or {}
            rec("   - gen_params 回显旋钮", gp.get("blend_steps") == 15 and abs(float(gp.get("blend_strength", 0)) - 0.65) < 1e-6,
                json.dumps(gp, ensure_ascii=False))
            images = j.get("images") or {}
            slots = list(images.keys())
            rec("   - 十槽位齐全", need.issubset(set(slots)),
                f"缺={sorted(need - set(slots))} 实有={len(slots)}")
            urls = collect_urls(images)
            nfilled = sum(1 for v in images.values() if v)
            rec("   - 十槽位已填充数 >= 9", nfilled >= 9,
                f"filled={nfilled}/10  " + json.dumps({k: (len(v) if isinstance(v, list) else v)
                                                       for k, v in images.items()}, ensure_ascii=False))
            rec("   - 有可访问图片 URL", len(urls) > 0, f"{len(urls)} 个")
            w = j.get("warnings") or []
            rec("   - 无管线告警", not w, f"{w}" if w else "clean")
            bad = []
            for slot, u in urls[:40]:
                p = u if u.startswith("/") else "/" + u.split("/", 3)[-1]
                c, ct, n = http_raw(p)
                if c != 200 or not n:
                    bad.append(f"{slot}:HTTP{c}")
            rec("   - 所有图片 URL 可回取", not bad, f"坏={bad[:5]}" if bad else f"{min(len(urls),40)} 个全 200")

        # ── 3. /api/regenerate 并发 2 请求 ──
        if sid is not None:
            res = {}
            def worker(i):
                t = time.time()
                try:
                    res[i] = http(f"/api/regenerate/{sid}",
                                  {"slot": "drone_raw", "prompt": f"[e2e-concurrent-{i}] ",
                                   "blend_steps": 10, "blend_strength": 0.5},
                                  timeout=900) + (time.time() - t,)
                except Exception as e:
                    res[i] = (0, {"error": repr(e)}, time.time() - t)
            ths = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
            t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]
            wall = time.time() - t
            oks = [k for k, v in res.items()
                   if v and v[0] == 200 and isinstance(v[1], dict) and v[1].get("ok")]
            rec("3. /api/regenerate 并发 x2 均成功", len(oks) == 2,
                f"wall={wall:.1f}s 各用时={[round(v[2], 1) for v in res.values()]}"
                + ("" if len(oks) == 2 else f"  err={[v[1] for v in res.values() if not isinstance(v[1], dict) or not v[1].get('ok')]}"))
            ssum = sum(v[2] for v in res.values())
            mx = max(v[2] for v in res.values())
            mn = min(v[2] for v in res.values())
            rec("   - 并发被串行化(排队者耗时≈2x 单人)", abs(wall - mx) <= 0.25 * mx and mx >= 1.5 * mn,
                f"wall={wall:.1f} max={mx:.1f} min={mn:.1f} (sum={ssum:.1f} 含等待，不可作判据)")

        # ── 4. /api/focus_drone ──
        if sid is not None:
            t = time.time()
            st, j = http(f"/api/focus_drone/{sid}",
                         {"nu": 0.55, "nv": 0.30, "target_px": 96, "blend_steps": 12,
                          "blend_strength": 0.6}, timeout=900)
            ok = st == 200 and isinstance(j, dict) and j.get("ok")
            rec("4. /api/focus_drone", ok, f"HTTP {st} {time.time()-t:.1f}s")
            if ok:
                imgs = j.get("images") or {}
                rec("   - 仍返回十槽位", need.issubset(set(imgs.keys())),
                    f"缺={sorted(need - set(imgs.keys()))}")

        # ── 5. 保存 + 读回 ──
        if sid is not None:
            st, j = http_multipart(f"/api/session/{sid}/images",
                                   {"final_rgb": "e2e_saved_final.png",
                                    "ir": "e2e_saved_ir.png"})
            ok1 = st == 200 and isinstance(j, dict) and bool(j.get("saved"))
            rec("5. /api/session/<sid>/images 保存(multipart)", ok1,
                f"HTTP {st} saved={list((j or {}).get('saved', {}).keys())}")
            st, j = http(f"/api/session/{sid}")
            sess = (j or {}).get("session") or {}
            nf = sum(1 for v in (sess.get("images") or {}).values() if v)
            rec("   - /api/session/<sid> 读回", st == 200 and "session" in (j or {}),
                f"HTTP {st} images_filled={nf}/10")
            st, j = http("/api/sessions")
            n = len((j or {}).get("sessions", [])) if isinstance(j, dict) else -1
            rec("   - /api/sessions 列表", st == 200 and n >= 1, f"n={n}")

    # ── 6. 收尾：关机 ──
    try:
        st, j = http("/api/shutdown", {})
        rec("6. /api/shutdown", st == 200, f"HTTP {st}")
    except Exception as e:
        # 进程在回包前自我 SIGTERM，连接被断属预期行为
        msg = f"{type(e).__name__}: {e}"
        rec("6. /api/shutdown", ("closed connection" in msg.lower()
                                 or "RemoteDisconnected" in msg
                                 or "ConnectionReset" in msg), msg)
    for _ in range(30):
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    rec("   - 进程已退出", proc.poll() is not None, f"code={proc.returncode}")

    # ── 7. 日志扫描 ──
    txt = open(log, "rb").read().decode("utf-8", "replace")
    bad = [pat for pat in ("Traceback (most recent call last)", "CUDA error", "RuntimeError:",
                           "Sizes of tensors must match", "Expected all tensors")
           if pat in txt]
    rec("7. 服务端日志无异常", not bad, f"命中={bad}" if bad else "clean")

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n===== {npass}/{len(RESULTS)} PASS =====", flush=True)
    with open(os.path.join(ROOT, "_e2e_report.json"), "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%F %T"), "pass": npass, "total": len(RESULTS),
                   "results": [{"name": n, "ok": o, "detail": d} for n, o, d in RESULTS]},
                  f, ensure_ascii=False, indent=2)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
