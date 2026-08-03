"""
Agent 1 JSON Schema — 语义描述规范 & DroneMMset manifest → Schema 映射
======================================================================
Transformer B 的输入源。定义 9 字段 JSON 结构，以及从 DroneMMset
manifest.jsonl 到该结构的完整映射逻辑。

使用:
    from json_schema import Agent1Schema, manifest_to_schema
    schema = Agent1Schema()
    json_obj = manifest_to_schema(manifest_line, schema)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


# ============================================================================
# 枚举定义
# ============================================================================

class DroneType(str, Enum):
    quadrotor = "quadrotor"


class DroneAction(str, Enum):
    hover = "hover"
    approach = "approach"       # 飞近
    retreat = "retreat"         # 飞远
    lateral_move = "lateral_move"  # 横向位移
    ascend = "ascend"
    descend = "descend"
    circle = "circle"           # 盘旋
    noise = "noise"             # 无规律微动


class TimeOfDay(str, Enum):
    dawn = "dawn"
    morning = "morning"
    afternoon = "afternoon"
    dusk = "dusk"
    night = "night"


class Weather(str, Enum):
    clear = "clear"
    overcast = "overcast"
    rainy = "rainy"
    foggy = "foggy"
    dusty = "dusty"
    backlight = "backlight"


class SceneType(str, Enum):
    urban = "urban"
    rural = "rural"
    mountain = "mountain"
    coastal = "coastal"
    desert = "desert"
    forest = "forest"
    industrial = "industrial"
    airfield = "airfield"


class CameraPosition(str, Enum):
    bottom = "bottom"   # 仰拍
    top = "top"         # 俯拍
    front = "front"     # 正面
    side = "side"       # 侧面
    back = "back"       # 背面


class Modality(str, Enum):
    RGB = "RGB"
    IR = "IR"


# ============================================================================
# 数据类 — Agent 1 JSON Schema 9字段
# ============================================================================

@dataclass
class TrajectoryPoint:
    """单个时间步的轨迹点"""
    t: float                     # 时间步序号
    action: DroneAction          # 该时刻的动作
    distance: float              # 距离（米）
    norm_u: float = 0.5          # 画面归一化水平位置 [0, 1]，0.5=中央
    norm_v: float = 0.5          # 画面归一化垂直位置 [0, 1]，0.5=中央

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "action": self.action.value,
            "distance": self.distance,
            "norm_u": self.norm_u,
            "norm_v": self.norm_v,
        }


@dataclass
class Camera:
    """相机参数"""
    position: CameraPosition
    elevation_deg: float = 0.0    # 仰角（度），bottom时为正
    fov_deg: float = 60.0         # 视场角（度）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.value,
            "elevation_deg": self.elevation_deg,
            "fov_deg": self.fov_deg,
        }


@dataclass
class Agent1Schema:
    """
    Agent 1 JSON Schema — 完整 9 字段语义描述
    
    这是 Transformer B 的唯一文本输入。每个字段编码了生成图像所需的
    一个语义维度，Transformer B 负责将语义映射为空间条件（depth + seg）。
    """
    drone_type: DroneType = DroneType.quadrotor
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    time_of_day: TimeOfDay = TimeOfDay.afternoon
    weather: Weather = Weather.clear
    scene_type: SceneType = SceneType.urban
    scene_description: str = ""
    modality: Modality = Modality.RGB
    camera: Camera = field(default_factory=Camera)
    confidence_note: str = ""  # LLM 低置信度附注

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_type": self.drone_type.value,
            "trajectory": [tp.to_dict() for tp in self.trajectory],
            "time_of_day": self.time_of_day.value,
            "weather": self.weather.value,
            "scene_type": self.scene_type.value,
            "scene_description": self.scene_description,
            "modality": self.modality.value,
            "camera": self.camera.to_dict(),
            "confidence_note": self.confidence_note,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_prompt(self) -> str:
        """
        将 Schema 转为自然语言 prompt，用于 ControlNet 条件文本输入。
        这是从结构化语义到自然语言的桥接。
        """
        parts = []
        # 场景
        parts.append(f"{self.time_of_day.value} {self.weather.value} weather, {self.scene_type.value} scene.")
        if self.scene_description:
            parts.append(f"Background: {self.scene_description}.")
        # 无人机
        parts.append(f"A {self.drone_type.value} drone.")
        # 轨迹
        if self.trajectory:
            actions = [tp.action.value for tp in self.trajectory]
            dists = [f"{tp.distance:.0f}m" for tp in self.trajectory]
            parts.append(f"Trajectory: {' → '.join(actions)}, distances: {', '.join(dists)}.")
        # 相机
        parts.append(f"C camera {self.camera.position.value} view, {self.camera.elevation_deg}° elevation, {self.camera.fov_deg}° FOV.")
        # 模态
        parts.append(f"Modality: {self.modality.value}.")
        return " ".join(parts)


# ============================================================================
# DroneMMset manifest.jsonl → Agent1Schema 映射
# ============================================================================

# 动作映射表（DroneMMset → Agent 1 DroneAction）
ACTION_MAP: Dict[str, DroneAction] = {
    "Hover":    DroneAction.hover,
    "Roll":     DroneAction.lateral_move,
    "Yaw":      DroneAction.circle,
    "Pitch":    DroneAction.approach,     # 默认 approach，需根据距离变化判断
    "Throttle": DroneAction.ascend,       # 默认 ascend，需根据距离变化判断
    "Noise":    DroneAction.noise,
}

# 无人机型号映射（DroneMMset → 通用标签）
DRONE_MODEL_MAP: Dict[str, str] = {
    "Air 2S":       "DJI Air 2S",
    "Mavic 2 Pro":   "DJI Mavic 2 Pro",
    "Mavic 3":       "DJI Mavic 3",
    "Mini 2":        "DJI Mini 2",
}

# 相机位置推断（基于 camera name 和 elevation）
# DroneMMset 有 Cam01、Cam02 两台相机，但具体位置需根据场景推断
# 这里用启发式规则


def infer_camera_position(camera_name: str, distance: float) -> CameraPosition:
    """
    根据相机名和距离推断拍摄位置。
    DroneMMset 中 Cam01/Cam02 为固定位置，具体朝向取决于场景。
    默认策略：距离 > 30m 视为地面拍摄（bottom），否则为正面。
    """
    # 启发式：远距离通常是仰拍，近距离正面
    if distance > 30:
        return CameraPosition.bottom
    elif distance > 10:
        return CameraPosition.front
    else:
        return CameraPosition.side


def infer_elevation(camera_name: str, distance: float) -> float:
    """推断仰角"""
    if distance > 50:
        return 45.0   # 远距离高仰角
    elif distance > 20:
        return 25.0
    else:
        return 10.0


def refine_action(action: DroneAction, distances: List[float], idx: int) -> DroneAction:
    """
    根据距离变化精修动作标签。
    Pitch:  距离递减 → approach, 递增 → retreat
    Throttle: 距离递增 → ascend, 递减 → descend
    """
    if idx == 0 or idx >= len(distances):
        return action

    delta = distances[idx] - distances[idx - 1]

    if action == DroneAction.approach:
        return DroneAction.approach if delta <= 0 else DroneAction.retreat
    elif action == DroneAction.ascend:
        return DroneAction.ascend if delta >= 0 else DroneAction.descend

    return action


def manifest_to_schema(manifest_entry: Dict[str, Any]) -> Agent1Schema:
    """
    将 DroneMMset manifest.jsonl 的一条记录映射为 Agent1Schema。
    
    Args:
        manifest_entry: manifest.jsonl 的一行（已解析为 dict）
    
    Returns:
        完整的 Agent1Schema 对象
    """
    # --- 基础属性 ---
    drone_model = manifest_entry.get("drone_model", "Unknown")
    distance_str = manifest_entry.get("distance", "50.0")
    distance = float(distance_str)
    action_name = manifest_entry.get("action", "Hover")
    camera_name = manifest_entry.get("camera", "Cam01")
    modality_str = manifest_entry.get("modality", "RGB")
    semantic_segments = manifest_entry.get("semantic_segments", [])
    
    # --- 场景描述 ---
    drone_full_name = DRONE_MODEL_MAP.get(drone_model, drone_model)
    scene_desc = f"{drone_full_name} drone at {distance}m distance, captured by {camera_name}."
    
    # --- 构建轨迹 ---
    trajectory: List[TrajectoryPoint] = []
    
    if semantic_segments:
        distances_for_refine: List[float] = []
        for seg in semantic_segments:
            raw_action = seg.get("action", "Hover")
            base_action = ACTION_MAP.get(raw_action, DroneAction.hover)
            distances_for_refine.append(distance)  # 同一帧内距离相同
        
        for i, seg in enumerate(semantic_segments):
            raw_action = seg.get("action", "Hover")
            ts = seg.get("ts", 0.0)
            base_action = ACTION_MAP.get(raw_action, DroneAction.hover)
            refined = refine_action(base_action, distances_for_refine, i)
            
            trajectory.append(TrajectoryPoint(
                t=ts,
                action=refined,
                distance=distance,
                norm_u=0.0,  # 标注数据中无此信息
                norm_v=0.0,
            ))
    else:
        # 单动作帧（无 semantic_segments）
        base_action = ACTION_MAP.get(action_name, DroneAction.hover)
        trajectory.append(TrajectoryPoint(
            t=0.0,
            action=base_action,
            distance=distance,
        ))
    
    # --- 相机 ---
    cam_position = infer_camera_position(camera_name, distance)
    elevation = infer_elevation(camera_name, distance)
    
    camera = Camera(
        position=cam_position,
        elevation_deg=elevation,
        fov_deg=60.0,
    )
    
    # --- 模态 ---
    modality = Modality.RGB if modality_str == "RGB" else Modality.IR
    
    # --- 时间/天气/场景（DroneMMset 未标注，使用合理默认值 + 启发式）---
    # 这些字段在 Layer 2 微调时需要从场景内容推断
    
    return Agent1Schema(
        drone_type=DroneType.quadrotor,
        trajectory=trajectory,
        time_of_day=TimeOfDay.afternoon,   # 默认（大部分录制在白天）
        weather=Weather.clear,              # 默认
        scene_type=SceneType.urban,         # 默认（大学校园场景）
        scene_description=scene_desc,
        modality=modality,
        camera=camera,
        confidence_note="Auto-mapped from DroneMMset manifest. time_of_day/weather/scene_type are defaults.",
    )


# ============================================================================
# 工具函数
# ============================================================================

def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """加载 manifest.jsonl 并返回 dict 列表"""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def batch_convert_manifest(manifest_path: str) -> List[Agent1Schema]:
    """批量转换整个 manifest.jsonl"""
    entries = load_manifest(manifest_path)
    return [manifest_to_schema(e) for e in entries]


# ============================================================================
# 示例运行
# ============================================================================
if __name__ == "__main__":
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
        "0-database/dronemmset/processed/manifest.jsonl"
    )
    
    entries = load_manifest(path)
    print(f"Loaded {len(entries)} entries from manifest.")
    
    # 展示前 3 条转换
    for i, entry in enumerate(entries[:3]):
        schema = manifest_to_schema(entry)
        print(f"\n{'='*60}")
        print(f"Entry {i+1}: {entry['frame_name']}")
        print(schema.to_json())
        print(f"\n→ Prompt: {schema.to_prompt()}")
