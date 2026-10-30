#!/usr/bin/env python3
"""Generate Agent 1 JSONs mimicking LLM output style for Phase 2 pairs.
Blends DroneMMset JSON structure with LLM-style natural language descriptions."""
import json, os, hashlib

BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
PAIRS = f"{BASE}/output/phase2_mixed/pairs_diverse.jsonl"
JSON_OUT = f"{BASE}/output/phase2_mixed/json_llm_style"
os.makedirs(JSON_OUT, exist_ok=True)

with open(PAIRS) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

def hash_int(name, lo, hi):
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return lo + h % (hi - lo)

def pick(name, lst):
    return lst[hash_int(name, 0, len(lst))]

DRONES = ["DJI Air 2S", "DJI Mavic 2 Pro", "DJI Mavic 3", "DJI Mini 2"]
ACTIONS = ["hover", "approach", "retreat", "lateral_move", "circle", "ascend", "descend"]
TOD = ["morning", "afternoon", "dusk"]
WEATHER = ["clear", "overcast"]
SCENES = {
    "sky": ["rural", "mountain", "coastal", "forest"],
    "ground": ["mountain", "rural", "forest", "farmland"],
}

def build_sky_json(name):
    drone = pick(name, DRONES)
    dist_start = float(hash_int(name, 30, 200))
    action = pick(name, ACTIONS)
    scene = pick(name, SCENES["sky"])
    tod_val = pick(name, TOD)
    weather_val = pick(name, WEATHER)
    elev = float(hash_int(name, 25, 65))
    is_ir = "_ir_" in name.lower()
    modality_val = "IR" if is_ir else "RGB"

    # LLM-style multi-step trajectory
    n_steps = 2 if hash_int(name, 0, 3) < 2 else 3
    traj = []
    dist = dist_start
    for i in range(n_steps):
        traj.append({
            "t": float(i),
            "action": action,
            "distance": round(dist, 1),
            "norm_u": round(0.3 + hash_int(name + str(i), 0, 41) / 100.0, 2),
            "norm_v": round(0.3 + hash_int(name + chr(65+i), 0, 41) / 100.0, 2),
        })
        dist = max(10, dist - 20 if action in ("approach",) else dist + 20 if action == "retreat" else dist)

    # Natural language scene_description (LLM style)
    desc = f"{tod_val} {weather_val}, {drone} drone {action} against open {scene} sky, "
    desc += f"distance {dist_start:.0f}m, bottom-up camera angle"

    return {
        "drone_type": "quadrotor",
        "trajectory": traj,
        "time_of_day": tod_val,
        "weather": weather_val,
        "scene_type": scene,
        "scene_description": desc,
        "modality": modality_val,
        "camera": {"position": "bottom", "elevation_deg": elev, "fov_deg": 60.0},
        "confidence_note": f"Auto-generated from {scene} sky scene template; drone model and distance are estimated.",
    }

def build_ground_json(name):
    drone = pick(name, DRONES)
    dist = float(hash_int(name, 20, 120))
    scene = pick(name, SCENES["ground"])
    tod_val = pick(name, TOD)
    weather_val = pick(name, WEATHER)
    elev = float(hash_int(name, 0, 15))
    action = "hover"

    traj = [{
        "t": 0.0, "action": action, "distance": round(dist, 1),
        "norm_u": round(0.3 + hash_int(name, 0, 41) / 100.0, 2),
        "norm_v": round(0.3 + hash_int(name + "v", 0, 41) / 100.0, 2),
    }]

    desc = f"{tod_val} {weather_val}, {drone} drone flying at {dist:.0f}m altitude, "
    desc += f"aerial top-down view of {scene} terrain"

    return {
        "drone_type": "quadrotor",
        "trajectory": traj,
        "time_of_day": tod_val,
        "weather": weather_val,
        "scene_type": scene,
        "scene_description": desc,
        "modality": "RGB",
        "camera": {"position": "top", "elevation_deg": elev, "fov_deg": 60.0},
        "confidence_note": f"Auto-generated from {scene} aerial terrain template; drone altitude estimated.",
    }

for p in pairs:
    name = p["frame_name"]
    scene = p["scene_type"]
    j = build_sky_json(name) if scene == "sky" else build_ground_json(name)
    json_path = f"{JSON_OUT}/{name}.json"
    with open(json_path, "w") as jf:
        json.dump(j, jf, indent=2)
    p["json_path"] = json_path

NEW = f"{BASE}/output/phase2_mixed/pairs_llm_style.jsonl"
with open(NEW, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(pairs)} LLM-style JSONs → {JSON_OUT}/")
print(f"Updated pairs → {NEW}")
