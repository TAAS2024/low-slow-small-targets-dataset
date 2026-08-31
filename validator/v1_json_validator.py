"""
S1 — Agent 1 (LLM) JSON 输出校验
==================================
输入端预校验第一关。纯规则引擎，零 GPU 消耗。

检查项（6 项，短路求值）：
  1. Schema 完整性      → A1_SCHEMA_INCOMPLETE
  2. 枚举值合法性        → A1_ENUM_INVALID
  3. 场景逻辑自洽        → A1_LOGIC_CONFLICT
  4. 描述关键词交叉检查  → A1_DESC_MISMATCH
  5. trajectory 值域    → A1_TRAJ_INVALID
  6. camera 参数合理区间 → A1_CAMERA_INVALID

用法:
    from v1_json_validator import S1JSONValidator
    v = S1JSONValidator(config_dir="config/")
    result = v.validate(json_data)
    # result = {"pass": True/False, "failure_code": "A1_XXX", "reason": "..."}
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ── 失败码常量 ──────────────────────────────────────────
A1_SCHEMA_INCOMPLETE   = "A1_SCHEMA_INCOMPLETE"
A1_ENUM_INVALID        = "A1_ENUM_INVALID"
A1_LOGIC_CONFLICT      = "A1_LOGIC_CONFLICT"
A1_DESC_MISMATCH       = "A1_DESC_MISMATCH"
A1_TRAJ_INVALID        = "A1_TRAJ_INVALID"
A1_CAMERA_INVALID      = "A1_CAMERA_INVALID"


class S1JSONValidator:
    """Agent 1 JSON 输出校验器 — 纯规则引擎"""

    # Agent 1 必须输出的 9 个字段
    REQUIRED_FIELDS = [
        "drone_type", "trajectory", "time_of_day", "weather",
        "scene_type", "scene_description", "modality", "camera", "confidence_note"
    ]

    # camera 字段的必填子字段
    REQUIRED_CAMERA_FIELDS = ["position", "fov_deg", "elevation_deg"]

    def __init__(self, config_dir: str = "config/"):
        """
        Args:
            config_dir: 配置文件目录，包含 ENUM_REGISTRY.yaml 和 exclusion_rules.yaml
        """
        self._enum = self._load_yaml(os.path.join(config_dir, "ENUM_REGISTRY.yaml"))
        self._rules = self._load_yaml(os.path.join(config_dir, "exclusion_rules.yaml"))

    # ── 公共入口 ─────────────────────────────────────────

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """短路求值：任一检查失败立即返回"""
        for check in [
            self._check_schema,
            self._check_enums,
            self._check_logic,
            self._check_description,
            self._check_trajectory,
            self._check_camera,
        ]:
            result = check(data)
            if not result["pass"]:
                return result
        return {"pass": True, "failure_code": None, "reason": "S1 全部 6 项检查通过"}

    # ── 检查 1: Schema 完整性 ────────────────────────────

    def _check_schema(self, data: Dict) -> Dict:
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            return self._fail(A1_SCHEMA_INCOMPLETE, f"缺少字段: {missing}")

        traj = data.get("trajectory")
        if not traj or not isinstance(traj, list) or len(traj) == 0:
            return self._fail(A1_SCHEMA_INCOMPLETE, "trajectory 为空或格式错误")

        cam = data.get("camera", {})
        cam_missing = [f for f in self.REQUIRED_CAMERA_FIELDS if f not in cam]
        if cam_missing:
            return self._fail(A1_SCHEMA_INCOMPLETE, f"camera 缺少子字段: {cam_missing}")

        return self._pass()

    # ── 检查 2: 枚举值合法性 ─────────────────────────────

    def _check_enums(self, data: Dict) -> Dict:
        checks = [
            ("drone_type", "drone_type"),
            ("weather", "weather"),
            ("time_of_day", "time_of_day"),
            ("scene_type", "scene_type"),
        ]
        for field, enum_key in checks:
            val = data.get(field)
            if val not in self._enum.get(enum_key, []):
                return self._fail(A1_ENUM_INVALID,
                    f"{field}='{val}' 不在合法枚举值 {self._enum.get(enum_key)} 中")

        cam_pos = data.get("camera", {}).get("position")
        if cam_pos not in self._enum.get("camera", {}).get("position", []):
            return self._fail(A1_ENUM_INVALID,
                f"camera.position='{cam_pos}' 不在合法枚举值中")

        return self._pass()

    # ── 检查 3: 场景逻辑自洽 ─────────────────────────────

    def _check_logic(self, data: Dict) -> Dict:
        for rule in self._rules.get("rules", []):
            match = True
            for field, expected in rule["fields"].items():
                # 特殊处理 elevation_deg_gt（大于比较）
                if field == "elevation_deg_gt":
                    actual = data.get("camera", {}).get("elevation_deg", -999)
                    if not (actual > expected):
                        match = False
                        break
                else:
                    # 支持嵌套字段如 camera.position
                    actual = self._nested_get(data, field)
                    if actual != expected:
                        match = False
                        break

            if match:
                verdict = rule["verdict"]
                if verdict == "REJECT":
                    return self._fail(A1_LOGIC_CONFLICT, rule["reason"])
                elif verdict == "WARN":
                    # WARN 不截停，但记录（后续可扩展为 warning 列表）
                    pass

        return self._pass()

    # ── 检查 4: scene_description 关键词交叉检查 ─────────

    def _check_description(self, data: Dict) -> Dict:
        desc = data.get("scene_description", "").lower()
        if not desc:
            return self._pass()

        for rule in self._rules.get("description_keywords", []):
            # 检查描述中是否包含任一关键词
            if any(kw.lower() in desc for kw in rule["keywords"]):
                conflict_field = rule["conflict_field"]
                actual_val = self._nested_get(data, conflict_field)
                if actual_val in rule["conflict_values"]:
                    return self._fail(A1_DESC_MISMATCH,
                        f"{rule['reason']}（description 含关键词，但 {conflict_field}='{actual_val}'）")

        return self._pass()

    # ── 检查 5: trajectory 值域 ──────────────────────────

    def _check_trajectory(self, data: Dict) -> Dict:
        traj = data.get("trajectory", [])
        valid_actions = self._enum.get("trajectory", {}).get("action", [])

        prev_t = -1.0
        for i, point in enumerate(traj):
            # distance >= 0
            if point.get("distance", -1) < 0:
                return self._fail(A1_TRAJ_INVALID,
                    f"trajectory[{i}].distance={point.get('distance')} < 0")

            # action 枚举合法
            action = point.get("action")
            if action and action not in valid_actions:
                return self._fail(A1_TRAJ_INVALID,
                    f"trajectory[{i}].action='{action}' 不在 {valid_actions} 中")

            # t 递增（如果有 t 字段）
            t = point.get("t")
            if t is not None:
                if t <= prev_t:
                    return self._fail(A1_TRAJ_INVALID,
                        f"trajectory[{i}].t={t} <= 前一帧 t={prev_t}（t 必须严格递增）")
                prev_t = t

        return self._pass()

    # ── 检查 6: camera 参数合理区间 ──────────────────────

    def _check_camera(self, data: Dict) -> Dict:
        cam = data.get("camera", {})
        fov = cam.get("fov_deg")
        fov_cfg = self._enum.get("camera", {}).get("fov_deg", {})

        if fov is not None:
            if fov < fov_cfg.get("min", 20) or fov > fov_cfg.get("max", 120):
                return self._fail(A1_CAMERA_INVALID,
                    f"fov_deg={fov} 超出 [{fov_cfg['min']}, {fov_cfg['max']}]")

        # elevation_deg 与 camera.position 的一致性
        position = cam.get("position")
        elevation = cam.get("elevation_deg")
        elev_ranges = self._enum.get("camera", {}).get("elevation_deg", {})

        if position and elevation is not None and position in elev_ranges:
            lo, hi = elev_ranges[position]
            if elevation < lo or elevation > hi:
                return self._fail(A1_CAMERA_INVALID,
                    f"camera.position='{position}' 但 elevation_deg={elevation} 超出 [{lo}, {hi}]")

        return self._pass()

    # ── 工具方法 ─────────────────────────────────────────

    def _nested_get(self, data: Dict, dotted_key: str) -> Any:
        """支持 'camera.position' 这样的嵌套键"""
        keys = dotted_key.split(".")
        val = data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return None
        return val

    def _fail(self, code: str, reason: str) -> Dict:
        return {"pass": False, "failure_code": code, "reason": reason}

    def _pass(self) -> Dict:
        return {"pass": True, "failure_code": None, "reason": ""}

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
