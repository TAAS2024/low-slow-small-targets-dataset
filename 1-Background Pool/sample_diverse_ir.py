#!/usr/bin/env python3
"""
IR 多样性采样脚本
策略：
  1. 每个视频源取 1 帧中间帧（跳过首尾过渡帧）
  2. 用感知哈希 (pHash) 去重，剔除跨视频的近似重复背景
  3. 最终精选 600 张 → IR_lora_training_samples_v2/
"""
import os, re, shutil
from collections import defaultdict
from PIL import Image
import imagehash  # pip install imagehash

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRS = {
    'DroneMMset': 'IR_raw_frames',
    'Anti-UAV': 'IR_raw_frames_antiuav',
}
OUTPUT_DIR = os.path.join(BASE, 'IR_lora_training_samples_v2')
TARGET_COUNT = 600

# ── Step 1: 按视频源分组 ──────────────────────────────────

def group_by_video(dir_name, pattern):
    """pattern: regex to extract (video_id, frame_num) from filename"""
    videos = defaultdict(list)
    for f in os.listdir(dir_name):
        if not f.endswith('.jpg'):
            continue
        m = re.match(pattern, f)
        if m:
            vid, frame = m.group(1), int(m.group(2))
            videos[vid].append((frame, f))
    return videos

drone_pattern = r'(.+)_(\d{6})\.jpg$'
antiuav_pattern = r'(.+)_(\d{4})\.jpg$'

drone_videos = group_by_video(os.path.join(BASE, SOURCE_DIRS['DroneMMset']), drone_pattern)
antiuav_videos = group_by_video(os.path.join(BASE, SOURCE_DIRS['Anti-UAV']), antiuav_pattern)

print(f"DroneMMset: {len(drone_videos)} videos, {sum(len(v) for v in drone_videos.values())} frames")
print(f"Anti-UAV:   {len(antiuav_videos)} videos, {sum(len(v) for v in antiuav_videos.values())} frames")

# ── Step 2: 每个视频取 1 帧中间帧 ──────────────────────────

def pick_middle_frame(videos_dict, source_dir, skip_head=2, skip_tail=2):
    """从每个视频取 1 帧中间帧，返回 [(source_dir, filename, video_id), ...]"""
    candidates = []
    for vid, frames in videos_dict.items():
        frames.sort(key=lambda x: x[0])  # sort by frame number
        if len(frames) <= skip_head + skip_tail:
            # 帧太少，取正中间
            idx = len(frames) // 2
        else:
            # 跳过首尾，取剩余部分的中间
            valid = frames[skip_head:-skip_tail]
            idx = len(valid) // 2
        frame_num, filename = frames[min(idx, len(frames)-1)]
        candidates.append((source_dir, filename, vid))
    return candidates

drone_candidates = pick_middle_frame(drone_videos, SOURCE_DIRS['DroneMMset'], skip_head=2, skip_tail=2)
antiuav_candidates = pick_middle_frame(antiuav_videos, SOURCE_DIRS['Anti-UAV'], skip_head=5, skip_tail=5)

all_candidates = drone_candidates + antiuav_candidates
print(f"\n候选帧: DroneMMset={len(drone_candidates)}, Anti-UAV={len(antiuav_candidates)}, 总计={len(all_candidates)}")

# ── Step 3: 感知哈希去重 ──────────────────────────────────

print("\n计算感知哈希...")
hashes = {}
for src_dir, fname, vid in all_candidates:
    path = os.path.join(BASE, src_dir, fname)
    try:
        img = Image.open(path).convert('L')  # 灰度，省计算
        h = imagehash.phash(img, hash_size=16)
        hashes[(src_dir, fname)] = h
    except Exception as e:
        print(f"  ⚠️ 无法读取 {fname}: {e}")

# 去重：汉明距离 ≤ 8 视为近似重复
THRESHOLD = 8
print(f"去重 (汉明距离 ≤ {THRESHOLD})...")

# 先按来源排序，优先保留 DroneMMset（视频间多样性更好）
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
    # 按来源比例抽样
    drone_selected = [(k, h) for (k, h) in selected if k[0] == SOURCE_DIRS['DroneMMset']]
    antiuav_selected = [(k, h) for (k, h) in selected if k[0] == SOURCE_DIRS['Anti-UAV']]
    
    # 维持比例（大致 1:1）
    n_drone = min(len(drone_selected), TARGET_COUNT // 2)
    n_antiuav = TARGET_COUNT - n_drone
    if n_antiuav > len(antiuav_selected):
        n_antiuav = len(antiuav_selected)
        n_drone = TARGET_COUNT - n_antiuav
    
    # 均匀间隔采样以保持多样性
    step_d = max(1, len(drone_selected) // n_drone)
    step_a = max(1, len(antiuav_selected) // n_antiuav)
    
    final = drone_selected[::step_d][:n_drone] + antiuav_selected[::step_a][:n_antiuav]
    # 去重（以防万一）
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
# 清空旧文件
for old in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, old))

copied = 0
for (src_dir, fname), _ in final:
    src = os.path.join(BASE, src_dir, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    shutil.copy2(src, dst)
    copied += 1

print(f"\n✅ 已复制 {copied} 帧到 {OUTPUT_DIR}")

# ── 统计 ──────────────────────────────────────────────────
d_count = sum(1 for (k, _) in final if k[0] == SOURCE_DIRS['DroneMMset'])
a_count = sum(1 for (k, _) in final if k[0] == SOURCE_DIRS['Anti-UAV'])
print(f"来源分布: DroneMMset={d_count}, Anti-UAV={a_count}")
