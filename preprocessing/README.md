# Data preparation

Two scripts, both free of model weights and both runnable on CPU:

| Script | Turns | Into |
|---|---|---|
| `prepare_background_pool.py` | a folder of photographs you have the right to use | the nine-category pool + `_metadata.jsonl` that `app/background_searcher.py` reads |
| `prepare_lora_dataset.py` | a folder of target images | a kohya-format training directory (images + caption sidecars) |

```bash
# background pool → 1-background-pool/curated_backgrounds/
python preprocessing/prepare_background_pool.py --src ~/photos --labels my_labels.jsonl

# target images → 0-database/kohya_dataset/10_drn3_uav/
python preprocessing/prepare_lora_dataset.py --src ~/drones \
    --out "$LSS_PROJECT_ROOT/0-database/kohya_dataset" --class-name drn3_uav
```

## Pool layout the retriever expects

```
1-background-pool/curated_backgrounds/
├── airfield/  coastal/  desert/  forest/  industrial/
├── mountain/  pure_sky/  rural/  urban/
│   ├── bg_<hash>.jpg
│   └── _metadata.jsonl
```

One JSONL row per image:

```json
{"filename": "bg_….jpg", "url": "…", "source": "…", "photographer": "…",
 "category": "urban", "time_of_day": "afternoon", "weather": "clear"}
```

`time_of_day ∈ {dawn, morning, afternoon, dusk, night}`,
`weather ∈ {clear, overcast, rainy, foggy, dusty, backlight}`. An unlabelled
image is not an error — it just only matches the retriever's fallback tier,
`match_level = "any"`. Labelling is what lets the four-tier fallback honour the
scene description.

Normalisation applied by the script: centre-crop to 16:9, resize to a common
width (default 1880 px), strip EXIF, reject near-duplicates by dHash. The paper's
pool is 37,852 images and is **not** distributed — the provenance fields exist so
that a pool can be audited, not so that it can be copied.

## Licence and privacy

The photographs keep their original terms. Keep `source`, `url` and
`photographer` populated, and check each source's terms before publishing a pool
or a model trained on it. EXIF, including GPS, is stripped on write.
