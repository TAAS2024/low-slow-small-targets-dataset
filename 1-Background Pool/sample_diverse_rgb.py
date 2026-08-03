#!/usr/bin/env python3
"""
RGB 多样性采样脚本（参照 IR 版本逻辑）
策略：
  1. 每个视频源取 1 帧中间帧（跳过首尾过渡帧）
  2. 用感知哈希 (pHash) 去重，剔除跨视频的近似重复背景
  3. 最终精选 600 张 → RGB_lora_training_samples_v2/
"""
import os, re, shutil
from collections import defaultdict
from PIL import Image
import imagehash

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRS = {
    'DroneMMset': 'RGB_raw_frames',
    'Anti-UAV': 'RGB_raw_frames_antiuav',
}
OUTPUT_DIR = os.path.join(BASE, 'RGB_lora_training_samples_v2')
TARGET_COUNT = 600

# ── Step 1: 按视频源分组 ──────────────────────────────────

def group_by_video(dir_name, pattern):
    videos = defaultdict(list)
    for f in os.listdir(dir_name):
        if not f.endswith('.jpg'):
            continue
        m = re.match(pattern, f)
        if m:
            vid, frame = m.group(1), int(m.group(2))
            videos[vid].append((frame, f))
    return videos

# DroneMMset RGB: drone_Cam01_Cam01-T0001-D00-A0001-S00_000025.jpg
drone_pattern = r'(.+)_(\d{6})\.jpg$'
# Anti-UAV RGB: 20190925_101846_1_1_0001.jpg (no "antiuav_" prefix)
antiuav_pattern = r'(.+)_(\d{4})\.jpg$'

drone_videos = group_by_video(os.path.join(BASE, SOURCE_DIRS['DroneMMset']), drone_pattern)
antiuav_videos = group_by_video(os.path.join(BASE, SOURCE_DIRS['Anti-UAV']), antiuav_pattern)

print(f"DroneMMset RGB: {len(drone_videos)} videos, {sum(len(v) for v in drone_videos.values())} frames")
print(f"Anti-UAV RGB:   {len(antiuav_videos)} videos, {sum(len(v) for v in antiuav_videos.values())} frames")

# ── Step 2: 每个视频取 1 帧中间帧 ──────────────────────────

def pick_middle_frame(videos_dict, source_dir, skip_head=2, skip_tail=2):
    candidates = []
    for vid, frames in videos_dict.items():
        frames.sort(key=lambda x: x[0])
        if len(frames) <= skip_head + skip_tail:
            idx = len(frames) // 2
        else:
            valid = frames[skip_head:-skip_tail]
            idx = len(valid) // 2
        frame_num, filename = valid[idx] if valid else frames[0]
        candidates.append((source_dir, filename, vid))
    return candidates

# RGB DroneMMset 帧间间隔大（25帧一跳），头尾跳 1 帧即可
drone_candidates = pick_middle_frame(drone_videos, SOURCE_DIRS['DroneMMset'], skip_head=1, skip_tail=1)
# RGB Anti-UAV 仍然是 1fps 连续帧，跳 5
antiuav_candidates = pick_middle_frame(antiuav_videos, SOURCE_DIRS['Anti-UAV'], skip_head=5, skip_tail=5)

all_candidates = drone_candidates + antiuav_candidates
print(f"\n候选帧: DroneMMset={len(drone_candidates)}, Anti-UAV={len(antiuav_candidates)}, 总计={len(all_candidates)}")

# ── Step 3: 感知哈希去重 ──────────────────────────────────

print("\n计算感知哈希...")
hashes = {}
for src_dir, fname, vid in all_candidates:
    path = os.path.join(BASE, src_dir, fname)
    try:
        img = Image.open(path).convert('L')
        h = imagehash.phash(img, hash_size=16)
        hashes[(src_dir, fname)] = h
    except Exception as e:
        print(f"  ⚠️ 无法读取 {fname}: {e}")

THRESHOLD = 8
print(f"去重 (汉明距离 ≤ {THRESHOLD})...")

def sort_key(item):
    src_dir, fname = item[0]
    return (0 if src_dir == SOURCE_DIRS['DroneMMset'] else 1, fname)

sorted_items = sorted(hashes.items(), key=sort_key)

selected = []
for (src_dir, fname), h in sorted_items:
    is_dup = False
    for _, sel_h in selected:
        if h - sel_h <= THRESHOLD:
            is_dup = True
            break
    if not is_dup:
        selected.append(((src_dir, fname), h))

print(f"去重后: {len(selected)} 帧")

# ── Step 4: 精选到目标数量 ────────────────────────────────

if len(selected) > TARGET_COUNT:
    drone_selected = [(k, h) for (k, h) in selected if k[0] == SOURCE_DIRS['DroneMMset']]
    antiuav_selected = [(k, h) for (k, h) in selected if k[0] == SOURCE_DIRS['Anti-UAV']]
    
    n_drone = min(len(drone_selected), TARGET_COUNT // 2)
    n_antiuav = TARGET_COUNT - n_drone
    if n_antiuav > len(antiuav_selected):
        n_antiuav = len(antiuav_selected)
        n_drone = TARGET_COUNT - n_antiuav
    
    step_d = max(1, len(drone_selected) // n_drone)
    step_a = max(1, len(antiuav_selected) // n_antiuav)
    
    final = drone_selected[::step_d][:n_drone] + antiuav_selected[::step_a][:n_antiuav]
    seen = set()
    unique_final = []
    for (k, h) in final:
        if k[1] not in seen:
            seen.add(k[1])
            unique_final.append((k, h))
    final = unique_final
else:
    final = selected

print(f"最终精选: {len(final)} 帧")

# ── Step 5: 复制到输出目录 ────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
for old in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, old))

copied = 0
for (src_dir, fname), _ in final:
    src = os.path.join(BASE, src_dir, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    shutil.copy2(src, dst)
    copied += 1

print(f"\n✅ 已复制 {copied} 帧到 {OUTPUT_DIR}")

d_count = sum(1 for (k, _) in final if k[0] == SOURCE_DIRS['DroneMMset'])
a_count = sum(1 for (k, _) in final if k[0] == SOURCE_DIRS['Anti-UAV'])
print(f"来源分布: DroneMMset={d_count}, Anti-UAV={a_count}")
