#!/usr/bin/env python3
"""Generate simplified background-only JSONs (no drone fields) for Layer 2 training."""
import json, os, hashlib

BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
PAIRS = f"{BASE}/output/phase2_mixed/pairs_diverse.jsonl"
JSON_OUT = f"{BASE}/output/phase2_mixed/json_bg_only"
os.makedirs(JSON_OUT, exist_ok=True)

with open(PAIRS) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

def h(name, lo, hi):
    return lo + int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (hi - lo)

def pick(name, lst):
    return lst[h(name, 0, len(lst))]

TOD = ["morning", "afternoon", "dusk"]
WEATHER = ["clear", "clear", "clear", "overcast"]
SCENES_SKY = ["rural", "mountain", "coastal", "forest"]
SCENES_GROUND = ["mountain", "rural", "forest", "farmland"]

SKY_DESCS = [
    "open sky with scattered clouds",
    "clear blue sky at high altitude",
    "overcast sky with uniform grey clouds",
    "bright sky with soft cloud formations",
    "wide open sky, horizon visible at the bottom",
    "dramatic sky with layered cloud banks",
]
GROUND_DESCS = [
    "aerial view of rolling mountain terrain",
    "top-down view of farmland with geometric field patterns",
    "bird's eye view of dense forest canopy",
    "aerial perspective of rural landscape with scattered trees",
    "overhead view of mountainous region with ridges and valleys",
    "satellite-style view of mixed terrain with patches of vegetation",
]

for p in pairs:
    name = p["frame_name"]
    scene = p["scene_type"]

    if scene == "sky":
        cam_pos, elev, scene_t = "bottom", float(h(name, 25, 65)), pick(name, SCENES_SKY)
        desc = pick(name, SKY_DESCS)
    else:
        cam_pos, elev, scene_t = "top", float(h(name, 0, 15)), pick(name, SCENES_GROUND)
        desc = pick(name, GROUND_DESCS)

    is_ir = "_ir_" in name.lower()
    j = {
        "time_of_day": pick(name, TOD),
        "weather": pick(name, WEATHER),
        "scene_type": scene_t,
        "scene_description": desc,
        "modality": "IR" if is_ir else "RGB",
        "camera": {"position": cam_pos, "elevation_deg": elev, "fov_deg": 60.0},
        "confidence_note": f"Background-only schema (no drone). Auto-generated from {scene} template.",
    }

    json_path = f"{JSON_OUT}/{name}.json"
    with open(json_path, "w") as jf:
        json.dump(j, jf, indent=2)
    p["json_path"] = json_path

NEW = f"{BASE}/output/phase2_mixed/pairs_bg_only.jsonl"
with open(NEW, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(pairs)} bg-only JSONs → {JSON_OUT}/")
print(f"Updated pairs → {NEW}")

# Show 2 samples
for scene in ["sky", "ground"]:
    samples = [p for p in pairs if p["scene_type"] == scene]
    p = samples[0]
    print(f"\n=== {scene}: {p['frame_name'][:45]} ===")
    with open(p["json_path"]) as jf:
        print(jf.read()[:300])
