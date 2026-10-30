"""
LLM Semantic Parser — Step 1 of the Generation Pipeline (v6.0)
==============================================================
将自然语言场景描述转换为双 JSON 输出：
  1. background_spec.json — 背景匹配专用（Agent 2 消费）
  2. full_scene_spec.json  — 全管线共享状态（Agent 3/4/5/6 消费）

输入示例:
  "阴天下午，城市高楼背景，四旋翼从下方仰拍"
  "晴天黄昏，四旋翼在城市上空从右往左横向飞过，正面拍摄"

输出: 两份 JSON（详见 extract_background_spec / to_full_spec 方法）
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Literal

# ============================================================
# 0. 默认配置
# ============================================================

DEFAULT_CONFIG = {
    "trajectory_action": "hover",
    "trajectory_distance": 75.0,
    "trajectory_norm_u": 0.5,
    "trajectory_norm_v": 0.3,       # 天空区域（画面上方）
    "scene_type": "puresky",
    "time_of_day": "afternoon",
    "weather": "clear",
    "camera_position": "bottom",
}

# ============================================================
# 1. 枚举词典
# ============================================================

class DroneAction(str, Enum):
    HOVER = "hover"                 # 悬停
    APPROACH = "approach"           # 飞近
    RETREAT = "retreat"             # 飞远
    LATERAL_MOVE = "lateral_move"   # 横向移动
    ASCEND = "ascend"               # 上升
    DESCEND = "descend"             # 下降
    CIRCLE_CW = "circle_cw"         # 顺时针盘旋
    CIRCLE_CCW = "circle_ccw"       # 逆时针盘旋


class Weather(str, Enum):
    CLEAR = "clear"                 # 晴天
    CLOUDY = "cloudy"               # 多云
    OVERCAST = "overcast"           # 阴天
    RAINY = "rainy"                 # 雨天
    FOGGY = "foggy"                 # 雾天
    SNOWY = "snowy"                 # 雪天
    DUSTY = "dusty"                 # 沙尘
    BACKLIGHT = "backlight"         # 逆光


class TimeOfDay(str, Enum):
    DAWN = "dawn"                   # 黎明
    MORNING = "morning"             # 上午
    NOON = "noon"                   # 正午
    AFTERNOON = "afternoon"         # 下午
    DUSK = "dusk"                   # 黄昏
    NIGHT = "night"                 # 夜间


class CameraPosition(str, Enum):
    """
    相机相对于无人机的位置。

    语义（以无人机为中心）：
      - bottom: 相机在下 → 仰拍 → 看到无人机腹部/底部
      - front:  相机在前 → 正前方视角
      - side:   相机在侧 → 侧面视角
      - back:   相机在后 → 尾部视角
      - top:    相机在上 → 俯拍 → 看到无人机顶部

    地面方向硬约定：所有视角下，无人机底座朝向画面下方（v=1.0 方向）。
    真实背景图的深度信息由 Agent 3/4 通过 DepthAnything V2 获取，
    不在此处用 elevation_deg / fov_deg 描述。
    """
    BOTTOM = "bottom"   # 仰拍（默认）
    FRONT = "front"     # 前方
    SIDE = "side"       # 侧面
    BACK = "back"       # 后方
    TOP = "top"         # 俯拍


class SceneType(str, Enum):
    PURESKY = "puresky"            # 纯天空（默认）
    URBAN = "urban"                # 城市
    RURAL = "rural"                # 乡村
    MOUNTAIN = "mountain"          # 山地
    COASTAL = "coastal"            # 海岸
    DESERT = "desert"              # 沙漠
    FOREST = "forest"              # 森林
    INDUSTRIAL = "industrial"      # 工业区
    AIRFIELD = "airfield"          # 机场
    SNOW = "snow"                  # 雪地
    NIGHT_CITY = "night_city"      # 城市夜景
    RESIDENTIAL = "residential"    # 住宅区


# ---- 语义映射表（中文关键词 → 枚举值） ----

ACTION_KEYWORDS = {
    "悬停": DroneAction.HOVER, "hover": DroneAction.HOVER,
    "飞近": DroneAction.APPROACH, "靠近": DroneAction.APPROACH, "接近": DroneAction.APPROACH,
    "approach": DroneAction.APPROACH,
    "飞远": DroneAction.RETREAT, "远离": DroneAction.RETREAT, "退后": DroneAction.RETREAT,
    "retreat": DroneAction.RETREAT,
    "横向": DroneAction.LATERAL_MOVE, "横移": DroneAction.LATERAL_MOVE, "平移": DroneAction.LATERAL_MOVE,
    "lateral": DroneAction.LATERAL_MOVE,
    "上升": DroneAction.ASCEND, "爬升": DroneAction.ASCEND, "升空": DroneAction.ASCEND,
    "ascend": DroneAction.ASCEND,
    "下降": DroneAction.DESCEND, "降低": DroneAction.DESCEND, "降落": DroneAction.DESCEND,
    "descend": DroneAction.DESCEND,
    "顺时针": DroneAction.CIRCLE_CW, "顺时针盘旋": DroneAction.CIRCLE_CW,
    "circle_cw": DroneAction.CIRCLE_CW,
    "逆时针": DroneAction.CIRCLE_CCW, "逆时针盘旋": DroneAction.CIRCLE_CCW,
    "盘旋": DroneAction.CIRCLE_CW, "绕圈": DroneAction.CIRCLE_CW, "环绕": DroneAction.CIRCLE_CW,
    "circle": DroneAction.CIRCLE_CW,
}

WEATHER_KEYWORDS = {
    "晴": Weather.CLEAR, "晴天": Weather.CLEAR, "晴朗": Weather.CLEAR,
    "clear": Weather.CLEAR, "sunny": Weather.CLEAR,
    "多云": Weather.CLOUDY, "cloudy": Weather.CLOUDY,
    "阴": Weather.OVERCAST, "阴天": Weather.OVERCAST,
    "overcast": Weather.OVERCAST,
    "雨": Weather.RAINY, "雨天": Weather.RAINY, "下雨": Weather.RAINY,
    "rain": Weather.RAINY, "rainy": Weather.RAINY,
    "雾": Weather.FOGGY, "雾天": Weather.FOGGY, "雾气": Weather.FOGGY,
    "fog": Weather.FOGGY, "foggy": Weather.FOGGY,
    "雪": Weather.SNOWY, "雪天": Weather.SNOWY, "下雪": Weather.SNOWY,
    "snow": Weather.SNOWY, "snowy": Weather.SNOWY,
    "沙尘": Weather.DUSTY, "沙": Weather.DUSTY, "尘": Weather.DUSTY,
    "dust": Weather.DUSTY, "sand": Weather.DUSTY,
    "逆光": Weather.BACKLIGHT, "背光": Weather.BACKLIGHT,
    "backlight": Weather.BACKLIGHT,
}

TIME_KEYWORDS = {
    "黎明": TimeOfDay.DAWN, "破晓": TimeOfDay.DAWN, "dawn": TimeOfDay.DAWN,
    "上午": TimeOfDay.MORNING, "早晨": TimeOfDay.MORNING, "早上": TimeOfDay.MORNING,
    "morning": TimeOfDay.MORNING,
    "正午": TimeOfDay.NOON, "中午": TimeOfDay.NOON, "noon": TimeOfDay.NOON,
    "下午": TimeOfDay.AFTERNOON, "午后": TimeOfDay.AFTERNOON,
    "afternoon": TimeOfDay.AFTERNOON,
    "黄昏": TimeOfDay.DUSK, "傍晚": TimeOfDay.DUSK, "日落": TimeOfDay.DUSK,
    "dusk": TimeOfDay.DUSK, "sunset": TimeOfDay.DUSK,
    "夜间": TimeOfDay.NIGHT, "夜晚": TimeOfDay.NIGHT, "晚上": TimeOfDay.NIGHT,
    "night": TimeOfDay.NIGHT,
}

CAMERA_KEYWORDS = {
    "下方": CameraPosition.BOTTOM, "仰拍": CameraPosition.BOTTOM,
    "底部": CameraPosition.BOTTOM, "bottom": CameraPosition.BOTTOM,
    "上方": CameraPosition.TOP, "俯拍": CameraPosition.TOP,
    "顶部": CameraPosition.TOP, "top": CameraPosition.TOP,
    "前方": CameraPosition.FRONT, "正面": CameraPosition.FRONT,
    "front": CameraPosition.FRONT,
    "侧面": CameraPosition.SIDE, "侧拍": CameraPosition.SIDE,
    "side": CameraPosition.SIDE,
    "后方": CameraPosition.BACK, "尾部": CameraPosition.BACK,
    "back": CameraPosition.BACK, "rear": CameraPosition.BACK,
}

SCENE_KEYWORDS = {
    "纯天空": SceneType.PURESKY, "天空": SceneType.PURESKY,
    "puresky": SceneType.PURESKY, "sky": SceneType.PURESKY,
    "城市": SceneType.URBAN, "都市": SceneType.URBAN, "高楼": SceneType.URBAN,
    "街道": SceneType.URBAN, "urban": SceneType.URBAN, "city": SceneType.URBAN,
    "乡村": SceneType.RURAL, "农村": SceneType.RURAL, "田野": SceneType.RURAL,
    "rural": SceneType.RURAL, "countryside": SceneType.RURAL,
    "山地": SceneType.MOUNTAIN, "山": SceneType.MOUNTAIN, "山脉": SceneType.MOUNTAIN,
    "mountain": SceneType.MOUNTAIN,
    "沿海": SceneType.COASTAL, "海岸": SceneType.COASTAL, "海": SceneType.COASTAL,
    "coastal": SceneType.COASTAL, "ocean": SceneType.COASTAL,
    "沙漠": SceneType.DESERT, "荒漠": SceneType.DESERT, "desert": SceneType.DESERT,
    "森林": SceneType.FOREST, "树林": SceneType.FOREST,
    "forest": SceneType.FOREST, "woods": SceneType.FOREST,
    "工业": SceneType.INDUSTRIAL, "工厂": SceneType.INDUSTRIAL,
    "industrial": SceneType.INDUSTRIAL, "factory": SceneType.INDUSTRIAL,
    "机场": SceneType.AIRFIELD, "跑道": SceneType.AIRFIELD,
    "airfield": SceneType.AIRFIELD, "airport": SceneType.AIRFIELD,
    "雪地": SceneType.SNOW, "雪景": SceneType.SNOW, "snow": SceneType.SNOW,
    "夜景": SceneType.NIGHT_CITY, "城市夜景": SceneType.NIGHT_CITY,
    "night_city": SceneType.NIGHT_CITY, "nightcity": SceneType.NIGHT_CITY,
    "住宅": SceneType.RESIDENTIAL, "小区": SceneType.RESIDENTIAL,
    "residential": SceneType.RESIDENTIAL,
}


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class TrajectoryPoint:
    """轨迹中的单个时间点"""
    t: float                  # 时间步序号（LLM 输出整数，JSON number 兼容 float）
    action: DroneAction       # 此时的动作
    distance: float           # 距相机距离（米）
    norm_u: float = 0.5       # 归一化水平位置 (0-1)
    norm_v: float = 0.3       # 归一化垂直位置 (0-1, 默认天空区域)


@dataclass
class CameraSpec:
    """相机参数（相对于无人机）— v6.0 精简，仅保留 position"""
    position: CameraPosition  # 相机相对无人机的位置

    # 以下字段已删除（v6.0）：
    #   elevation_deg — 背景是真实图片，透视由图片本身决定
    #   fov_deg       — 同上
    #
    # 地面方向硬约定：
    #   无人机底座始终朝向画面下方（v=1.0）。所有 camera position
    #   （bottom/front/side/back/top）均适用。Agent 4 Transformer
    #   渲染时直接硬编码此约定。


@dataclass
class SceneSpec:
    """Step 1 输出：完整场景描述（内部表示）"""
    # 轨迹
    trajectory: list[TrajectoryPoint]

    # 环境属性
    time_of_day: TimeOfDay
    weather: Weather
    scene_type: SceneType
    scene_description: str = ""

    # 相机
    camera: CameraSpec = field(default_factory=lambda: CameraSpec(position=CameraPosition.BOTTOM))

    # 元数据
    raw_input: str = ""
    confidence_note: str = ""

    # 以下字段已删除（v6.0）：
    #   drone_type — 固定四旋翼（LoRA 仅有四旋翼数据）
    #   modality   — 始终 RGB→IR 转换，不在此处决策

    # ── 双 JSON 输出方法 ──

    def extract_background_spec(self) -> dict:
        """
        输出 background_spec.json（Agent 2 消费）。
        仅含背景匹配所需字段：scene_type + time_of_day + weather + camera_position
        """
        return {
            "scene_type": self.scene_type.value,
            "time_of_day": self.time_of_day.value,
            "weather": self.weather.value,
            "camera_position": self.camera.position.value,
        }

    def to_full_spec(self) -> dict:
        """
        输出 full_scene_spec.json（Agent 3/4/5/6 消费）。
        含无人机轨迹 + 背景 + 相机 + 元数据，供全管线使用。
        """
        traj_list = []
        for pt in self.trajectory:
            traj_list.append({
                "t": pt.t,
                "action": pt.action.value,
                "distance": pt.distance,
                "norm_u": pt.norm_u,
                "norm_v": pt.norm_v,
            })

        return {
            "trajectory": traj_list,
            "background": {
                "scene_type": self.scene_type.value,
                "time_of_day": self.time_of_day.value,
                "weather": self.weather.value,
                "scene_description": self.scene_description,
            },
            "camera": {
                "position": self.camera.position.value,
            },
            "meta": {
                "raw_input": self.raw_input,
                "confidence_note": self.confidence_note,
                "default_fills": self._detect_default_fills(),
            },
        }

    def _detect_default_fills(self) -> list[str]:
        """检测哪些字段用的是默认值（用户未提供）。"""
        fills = []
        if self.time_of_day == TimeOfDay.AFTERNOON:
            fills.append("time_of_day=afternoon")
        if self.weather == Weather.CLEAR:
            fills.append("weather=clear")
        if self.scene_type == SceneType.PURESKY:
            fills.append("scene_type=puresky")
        if self.camera.position == CameraPosition.BOTTOM:
            fills.append("camera.position=bottom")
        if len(self.trajectory) == 1:
            tp = self.trajectory[0]
            if tp.action == DroneAction.HOVER:
                fills.append("trajectory.action=hover")
            if tp.distance == DEFAULT_CONFIG["trajectory_distance"]:
                fills.append(f"trajectory.distance={tp.distance}m")
            if tp.norm_u == DEFAULT_CONFIG["trajectory_norm_u"] and tp.norm_v == DEFAULT_CONFIG["trajectory_norm_v"]:
                fills.append(f"trajectory.norm_u/v={tp.norm_u}/{tp.norm_v}")
        return fills

    def to_background_json(self, indent: int = None) -> str:
        """background_spec.json 的 JSON 字符串。"""
        return json.dumps(self.extract_background_spec(), ensure_ascii=False, indent=indent)

    def to_full_json(self, indent: int = 2) -> str:
        """full_scene_spec.json 的 JSON 字符串。"""
        return json.dumps(self.to_full_spec(), ensure_ascii=False, indent=indent)


# ============================================================
# 3. Few-shot Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个无人机场景解析器。将用户对无人机侦察场景的自然语言描述精确转换为 JSON。

## 重要：双 JSON 输出模式

你会生成两份 JSON，但**只输出 full_scene_spec**。background_spec 由代码自动从 full_scene_spec 提取。

## 字段说明

| 字段 | 类型 | 可选值 | 默认 | 说明 |
|:--|:--|:--|:--|:--|
| trajectory | array | — | [{"t":0,"action":"hover","distance":75,"norm_u":0.5,"norm_v":0.3}] | 逐时间步轨迹。默认悬停75m，无人机在天空区域(画面偏上,norm_v=0.3) |
| trajectory[].action | string | hover/approach/retreat/lateral_move/ascend/descend/circle_cw/circle_ccw | hover | 飞行动作 |
| trajectory[].distance | float | 20-500 | 75 | 距相机距离(米) |
| trajectory[].norm_u | float | 0.0-1.0 | 0.5 | 画面水平位置 |
| trajectory[].norm_v | float | 0.0-1.0 | 0.3 | 画面垂直位置(0=上,1=下,默认天空区域) |
| time_of_day | string | dawn/morning/noon/afternoon/dusk/night | afternoon | 时段 |
| weather | string | clear/cloudy/overcast/rainy/foggy/snowy/dusty/backlight | clear | 天气 |
| scene_type | string | puresky/urban/rural/mountain/coastal/desert/forest/industrial/airfield/snow/night_city/residential | puresky | 背景场景类型 |
| scene_description | string | 自由文本 | "" | 背景场景的简短英文描述 |
| camera.position | string | bottom/front/side/back/top | bottom | 相机相对于无人机的位置 |
| confidence_note | string | "" | "" | 低置信度推断标注 |

## 相机位置语义

以无人机为参照中心：
- **bottom**: 相机在无人机下方 → 仰拍 → 看见无人机腹部/底部（默认）
- **front**: 相机在无人机前方 → 正面视角
- **side**: 相机在无人机侧面 → 侧面视角
- **back**: 相机在无人机后方 → 尾部视角
- **top**: 相机在无人机上方 → 俯拍 → 看见无人机顶部

## 默认规则

- 未指定的字段使用默认值，在 confidence_note 中标注
- 轨迹默认：悬停(hover) + 75m + 画面偏上天空区域(norm_u=0.5, norm_v=0.3)
- 背景默认：纯天空(puresky) + 下午(afternoon) + 晴天(clear)
- 相机默认：仰拍(bottom)
- 无人机类型固定为四旋翼（不输出此字段）
- 输出只包含 JSON，不要有其他文字，不要 markdown 代码块标记

## 地面方向硬约定

所有视角下，无人机底座朝向画面下方。不在此处描述 elevation_deg / fov_deg。

## 示例

输入: "阴天下午，城市高楼背景，从下方仰拍"
输出:
{"trajectory":[{"t":0,"action":"hover","distance":75.0,"norm_u":0.5,"norm_v":0.3}],"time_of_day":"afternoon","weather":"overcast","scene_type":"urban","scene_description":"city high-rise buildings under overcast sky","camera":{"position":"bottom"},"confidence_note":""}

输入: "晴天黄昏，四旋翼在城市上空从右往左横向飞过，正面拍摄"
输出:
{"trajectory":[{"t":0,"action":"lateral_move","distance":80.0,"norm_u":0.3,"norm_v":0.5},{"t":1,"action":"lateral_move","distance":80.0,"norm_u":0.7,"norm_v":0.5}],"time_of_day":"dusk","weather":"clear","scene_type":"urban","scene_description":"sunset over city skyline with golden light","camera":{"position":"front"},"confidence_note":""}

输入: "雾天黎明，森林上空盘旋，距离约100米，侧面拍摄"
输出:
{"trajectory":[{"t":0,"action":"hover","distance":100.0,"norm_u":0.5,"norm_v":0.4}],"time_of_day":"dawn","weather":"foggy","scene_type":"forest","scene_description":"dense forest canopy in misty dawn atmosphere","camera":{"position":"side"},"confidence_note":"悬停替代盘旋(距离远时盘旋视觉效果近似)"}

输入: "大晴天上午，沙漠地带，四旋翼从100米远处飞近到50米"
输出:
{"trajectory":[{"t":0,"action":"approach","distance":100.0,"norm_u":0.5,"norm_v":0.5},{"t":1,"action":"approach","distance":50.0,"norm_u":0.5,"norm_v":0.5}],"time_of_day":"morning","weather":"clear","scene_type":"desert","scene_description":"vast desert landscape under bright morning sun","camera":{"position":"bottom"},"confidence_note":"未指定视角，默认仰拍"}

输入: "雨天正午，工业区背景，无人机从200米远处飞近到50米，尾部视角"
输出:
{"trajectory":[{"t":0,"action":"approach","distance":200.0,"norm_u":0.5,"norm_v":0.5},{"t":1,"action":"approach","distance":50.0,"norm_u":0.5,"norm_v":0.5}],"time_of_day":"noon","weather":"rainy","scene_type":"industrial","scene_description":"industrial complex with factories under rainy sky","camera":{"position":"back"},"confidence_note":""}

输入: "雪天黄昏，住宅区，横向移动从左侧飞到右侧，俯拍"
输出:
{"trajectory":[{"t":0,"action":"lateral_move","distance":100.0,"norm_u":0.2,"norm_v":0.5},{"t":1,"action":"lateral_move","distance":100.0,"norm_u":0.8,"norm_v":0.5}],"time_of_day":"dusk","weather":"snowy","scene_type":"residential","scene_description":"suburban residential area covered in snow at dusk","camera":{"position":"top"},"confidence_note":"俯拍时无人机在画面偏上方"}
"""


# ============================================================
# 4. JSON 解析与校验
# ============================================================

def validate_and_parse(raw_json: str, raw_input: str = "") -> SceneSpec:
    """
    将 LLM 原始输出校验并解析为 SceneSpec。

    容忍常见问题：markdown 代码块包裹、尾部逗号、枚举值别名等。
    """
    # 剥离 markdown 代码块
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # 修复常见格式问题
    cleaned = re.sub(r",\s*}", "}", cleaned)   # 尾部逗号
    cleaned = re.sub(r",\s*]", "]", cleaned)   # 数组尾部逗号

    data = json.loads(cleaned)

    # 解析 trajectory
    traj_list = []
    for pt in data.get("trajectory", []):
        action_str = pt.get("action", "hover")
        action = _coerce_enum(action_str, DroneAction, ACTION_KEYWORDS, "hover")
        traj_list.append(TrajectoryPoint(
            t=pt.get("t", 0),
            action=action,
            distance=float(pt.get("distance", DEFAULT_CONFIG["trajectory_distance"])),
            norm_u=float(pt.get("norm_u", DEFAULT_CONFIG["trajectory_norm_u"])),
            norm_v=float(pt.get("norm_v", DEFAULT_CONFIG["trajectory_norm_v"])),
        ))

    # 至少一个轨迹点
    if not traj_list:
        traj_list = [TrajectoryPoint(
            t=0,
            action=DroneAction.HOVER,
            distance=DEFAULT_CONFIG["trajectory_distance"],
            norm_u=DEFAULT_CONFIG["trajectory_norm_u"],
            norm_v=DEFAULT_CONFIG["trajectory_norm_v"],
        )]

    # 解析 camera
    cam_data = data.get("camera", {})
    cam_pos = _coerce_enum(
        cam_data.get("position", DEFAULT_CONFIG["camera_position"]),
        CameraPosition, CAMERA_KEYWORDS, DEFAULT_CONFIG["camera_position"]
    )

    spec = SceneSpec(
        trajectory=traj_list,
        time_of_day=_coerce_enum(
            data.get("time_of_day", DEFAULT_CONFIG["time_of_day"]),
            TimeOfDay, TIME_KEYWORDS, DEFAULT_CONFIG["time_of_day"]
        ),
        weather=_coerce_enum(
            data.get("weather", DEFAULT_CONFIG["weather"]),
            Weather, WEATHER_KEYWORDS, DEFAULT_CONFIG["weather"]
        ),
        scene_type=_coerce_enum(
            data.get("scene_type", DEFAULT_CONFIG["scene_type"]),
            SceneType, SCENE_KEYWORDS, DEFAULT_CONFIG["scene_type"]
        ),
        scene_description=data.get("scene_description", ""),
        camera=CameraSpec(position=cam_pos),
        raw_input=raw_input,
        confidence_note=data.get("confidence_note", ""),
    )
    return spec


def _coerce_enum(value: str, enum_cls, keyword_map: dict, default: str):
    """将字符串强制转为枚举值，支持关键词映射。"""
    if isinstance(value, enum_cls):
        return value
    v = str(value).strip().lower()
    # 直接匹配枚举值（大小写不敏感）
    for e in enum_cls:
        if e.value.lower() == v:
            return e
    # 关键词映射
    if v in keyword_map:
        return keyword_map[v]
    # 部分匹配
    for kw, ev in keyword_map.items():
        if kw in v or v in kw:
            return ev
    # 回退
    return enum_cls(default)


# ============================================================
# 5. LLM 调用接口（框架无关）
# ============================================================

def build_messages(user_input: str) -> list[dict]:
    """构建 LLM 消息列表（兼容 OpenAI/Anthropic 等格式）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input.strip()},
    ]


def parse_with_openai(user_input: str, model: str = "gpt-4o", api_key: str = None) -> SceneSpec:
    """
    使用 OpenAI API 解析。需要 openai 包：pip install openai
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    messages = build_messages(user_input)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content
    return validate_and_parse(raw, raw_input=user_input)


def parse_with_claude(user_input: str, model: str = "claude-sonnet-4-20250514", api_key: str = None) -> SceneSpec:
    """
    使用 Anthropic Claude API 解析。需要 anthropic 包：pip install anthropic
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("请安装 anthropic 包: pip install anthropic")

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    messages = build_messages(user_input)
    response = client.messages.create(
        model=model,
        system=messages[0]["content"],
        messages=messages[1:],
        temperature=0.1,
        max_tokens=1000,
    )
    raw = response.content[0].text
    return validate_and_parse(raw, raw_input=user_input)


def parse_with_deepseek(user_input: str, model: str = "deepseek-chat", api_key: str = None) -> SceneSpec:
    """
    使用 DeepSeek API 解析（兼容 OpenAI SDK，仅 base_url 不同）。
    需要 openai 包：pip install openai
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    client = OpenAI(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    messages = build_messages(user_input)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content
    return validate_and_parse(raw, raw_input=user_input)


# ============================================================
# 6. 便捷函数：一步解析 + 双 JSON 输出
# ============================================================

def parse(user_input: str, backend: Literal["openai", "claude", "deepseek", "dry"] = "dry",
          model: str = "", api_key: str = "") -> SceneSpec:
    """
    一步解析用户输入。

    Args:
        user_input: 自然语言场景描述
        backend: "openai" | "claude" | "deepseek" | "dry"（dry-run 返回默认值用于测试）
        model: 模型名（空则用默认值）
        api_key: API key（空则从环境变量取）

    Returns:
        SceneSpec: 结构化场景参数
    """
    if backend == "dry":
        # dry-run: 不调 LLM，返回带有 raw_input 的默认 spec
        return SceneSpec(
            trajectory=[TrajectoryPoint(
                t=0,
                action=DroneAction.HOVER,
                distance=DEFAULT_CONFIG["trajectory_distance"],
                norm_u=DEFAULT_CONFIG["trajectory_norm_u"],
                norm_v=DEFAULT_CONFIG["trajectory_norm_v"],
            )],
            time_of_day=TimeOfDay.AFTERNOON,
            weather=Weather.CLEAR,
            scene_type=SceneType.PURESKY,
            raw_input=user_input,
            confidence_note="[dry-run] 未调用 LLM，使用默认值",
        )
    elif backend == "openai":
        return parse_with_openai(user_input, model=model or "gpt-4o", api_key=api_key)
    elif backend == "claude":
        return parse_with_claude(user_input, model=model or "claude-sonnet-4-20250514", api_key=api_key)
    elif backend == "deepseek":
        return parse_with_deepseek(user_input, model=model or "deepseek-chat", api_key=api_key)
    else:
        raise ValueError(f"不支持的 backend: {backend}")


def parse_to_dual_json(user_input: str, backend: str = "dry", **kwargs) -> dict:
    """
    一步解析 + 输出双 JSON。

    Returns:
        {
            "background_spec": {...},   # Agent 2 消费
            "full_scene_spec": {...},   # Agent 3/4/5/6 消费
            "background_json": "...",   # JSON string
            "full_json": "...",         # JSON string
        }
    """
    spec = parse(user_input, backend=backend, **kwargs)
    return {
        "background_spec": spec.extract_background_spec(),
        "full_scene_spec": spec.to_full_spec(),
        "background_json": spec.to_background_json(indent=2),
        "full_json": spec.to_full_json(indent=2),
    }


# ============================================================
# 8. CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM Parser v6.0: 自然语言 → 双 JSON")
    p.add_argument("input", nargs="?", help="场景描述文本，不提供则进入交互模式")
    p.add_argument("--backend", choices=["openai", "claude", "deepseek", "dry"], default="dry",
                   help="LLM 后端 (default: dry)")
    p.add_argument("--model", default="", help="覆盖默认模型")
    p.add_argument("--api-key", default="", help="API key（或设环境变量）")
    p.add_argument("--dual", action="store_true", default=True,
                   help="输出双 JSON 格式 (默认开启)")
    args = p.parse_args()

    if args.input:
        if args.dual:
            result = parse_to_dual_json(args.input, backend=args.backend, model=args.model, api_key=args.api_key)
            print("=" * 50)
            print("📦 background_spec.json  →  Agent 2 背景匹配")
            print("=" * 50)
            print(result["background_json"])
            print()
            print("=" * 50)
            print("📦 full_scene_spec.json  →  Agent 3/4/5/6 全管线")
            print("=" * 50)
            print(result["full_json"])
        else:
            spec = parse(args.input, backend=args.backend, model=args.model, api_key=args.api_key)
            print(spec.to_full_json())
    else:
        # 交互模式
        print("LLM Parser v6.0 交互模式 (Ctrl+C 退出)")
        print(f"后端: {args.backend}  |  模式: 双 JSON")
        print("-" * 50)
        try:
            while True:
                user = input("\n场景描述 > ").strip()
                if not user:
                    continue
                result = parse_to_dual_json(user, backend=args.backend, model=args.model, api_key=args.api_key)
                print()
                print("📦 background_spec.json:")
                print(result["background_json"])
                print()
                print("📦 full_scene_spec.json:")
                print(result["full_json"])
                print("-" * 50)
        except KeyboardInterrupt:
            print("\n退出。")
