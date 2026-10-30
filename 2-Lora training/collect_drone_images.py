"""
无人机图片采集 — Unsplash Source (免费, 无需 API Key)
=====================================================
"""
import requests, hashlib, time, io
from pathlib import Path
from PIL import Image
import numpy as np

OUTPUT_DIR = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/0-database/11_drone_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Unsplash Source: 每次返回随机图片
# 多关键词组合覆盖不同场景
QUERIES = [
    "drone-quadcopter-product",
    "drone-white-background",
    "DJI-drone",
    "quadcopter-isolated",
    "drone-top-view",
    "drone-flying",
    "drone-sky",
    "drone-studio",
    "drone-professional",
    "drone-commercial",
]

TOTAL_WANTED = 150
PER_QUERY = 20
seen_hashes = set()
total = 0

for query in QUERIES:
    tag = query.replace("-", "_")
    downloaded = 0

    for i in range(PER_QUERY):
        url = f"https://source.unsplash.com/800x800/?{query}"
        try:
            r = requests.get(url, timeout=15, allow_redirects=True)
            if r.status_code != 200:
                continue

            # 获取实际图片 URL（Unsplash 会重定向）
            final_url = r.url

            # 通过 hash 去重
            h = hashlib.md5(r.content).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # 验证
            img = Image.open(io.BytesIO(r.content))
            w, h_sz = img.size
            if w < 200 or h_sz < 200:
                continue
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            if arr.std() < 8:
                continue

            fname = f"{tag}_{downloaded:04d}_{h[:8]}.jpg"
            with open(OUTPUT_DIR / fname, "wb") as f:
                f.write(r.content)
            downloaded += 1
            total += 1
            if total % 10 == 0:
                print(f"   📥 {total} 张...")

        except Exception as e:
            pass
        time.sleep(0.4)

    print(f"   ✅ {tag}: {downloaded} 张")

print(f"\n✅ 完成！共 {total} 张 → {OUTPUT_DIR}")
