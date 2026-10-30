#!/usr/bin/env python3
"""
template_parser.py — deterministic template/keyword baseline for Module M1
=========================================================================
Purpose
-------
Major Concern 11 of the review asks for a comparison between the LLM parser and
"a deterministic template-based parser".  This module implements that baseline:
an exhaustive, longest-match keyword parser that consumes *exactly the same
controlled vocabulary* as the deployed semantic parser (ACTION_KEYWORDS,
WEATHER_KEYWORDS, TIME_KEYWORDS, CAMERA_KEYWORDS, SCENE_KEYWORDS imported from
``3-LLM starter/llm_parser.py``), plus a numeric template for distance phrases
("from 100 m to 300 m", "悬停在 75 米").

It is deterministic: identical input always yields identical output, no network
access, no sampling, and it cannot generalise beyond its vocabulary.  This is
precisely the trade-off the benchmark is designed to expose.

Output schema mirrors the fields scored in the benchmark:
    scene_type, time_of_day, weather, camera_position, action,
    distance_first, distance_last, matched (list of retrieved fields)
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the controlled vocabulary from the deployed parser (single source of truth)
# ---------------------------------------------------------------------------
_PARSER_PATH = (Path(__file__).resolve().parents[2]
                / "3-LLM starter" / "llm_parser.py")

_spec = importlib.util.spec_from_file_location("llm_parser_ref", _PARSER_PATH)
llm_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_parser)

ACTION_KEYWORDS = llm_parser.ACTION_KEYWORDS
WEATHER_KEYWORDS = llm_parser.WEATHER_KEYWORDS
TIME_KEYWORDS = llm_parser.TIME_KEYWORDS
CAMERA_KEYWORDS = llm_parser.CAMERA_KEYWORDS
SCENE_KEYWORDS = llm_parser.SCENE_KEYWORDS
DEFAULT_CONFIG = llm_parser.DEFAULT_CONFIG

_FIELDS = [
    ("scene_type", SCENE_KEYWORDS, DEFAULT_CONFIG["scene_type"]),
    ("time_of_day", TIME_KEYWORDS, DEFAULT_CONFIG["time_of_day"]),
    ("weather", WEATHER_KEYWORDS, DEFAULT_CONFIG["weather"]),
    ("camera_position", CAMERA_KEYWORDS, DEFAULT_CONFIG["camera_position"]),
]

_NUM_ZH = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _longest_match(text: str, table: dict) -> tuple[str | None, str | None]:
    """Return (keyword, enum_value) for the longest keyword occurring in text."""
    low = text.lower()
    best, best_key = None, None
    for kw in table:
        k = kw.lower()
        pos = low.find(k)
        if pos < 0:
            continue
        # prefer the longest keyword; break ties by earliest occurrence
        key = (len(k), -pos)
        if best_key is None or key > best_key:
            best, best_key = k, key
    if best is None:
        return None, None
    return best, table[best].value if hasattr(table[best], "value") else table[best]


def _numbers(text: str) -> list[float]:
    """Extract metric distances (metres) from the description, in order."""
    vals: list[float] = []
    # Arabic numerals immediately followed/accompanied by a metric unit
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:米|m\b|metres|meters)", text, re.I):
        vals.append(float(m.group(1)))
    if vals:
        return vals
    # Chinese numerals, e.g. "一百米"
    for m in re.finditer(r"([零一二两三四五六七八九十]{1,3})\s*(?:米|m\b)", text):
        s = m.group(1)
        if s == "十":
            v = 10
        elif "十" in s:
            a, _, b = s.partition("十")
            v = (_NUM_ZH.get(a, 1) * 10) + (_NUM_ZH.get(b, 0) if b else 0)
        else:
            v = _NUM_ZH.get(s, 0)
        if v:
            vals.append(float(v))
    return vals


def parse(text: str) -> dict:
    """Deterministic parse of a scene description."""
    out: dict = {}
    matched: list[str] = []
    for field, table, default in _FIELDS:
        kw, val = _longest_match(text, table)
        out[field] = val if val is not None else default
        if val is not None:
            matched.append(field)

    kw, action = _longest_match(text, ACTION_KEYWORDS)
    out["action"] = action if action is not None else "hover"
    if action is not None:
        matched.append("action")

    nums = _numbers(text)
    if nums:
        out["distance_first"] = nums[0]
        out["distance_last"] = nums[-1]
        out["n_points"] = 2 if len(nums) > 1 else 1
    else:
        out["distance_first"] = DEFAULT_CONFIG["trajectory_distance"]
        out["distance_last"] = DEFAULT_CONFIG["trajectory_distance"]
        out["n_points"] = 1

    out["matched"] = matched
    # every field that no keyword populated is a silently defaulted field
    out["silent_defaults"] = [f for f, _, _ in _FIELDS if f not in matched] + \
                             ([] if "action" in matched else ["action"])
    out["raw_output"] = ""
    return out


if __name__ == "__main__":
    for t in ["晴天下午，城市高楼背景，四旋翼悬停在天空中",
              "天刚擦黑，摩天楼群之间，飞行器悬停不动",
              "无人机从100米处飞远到300米"]:
        print(t, "->", {k: v for k, v in parse(t).items() if k != "matched"})
