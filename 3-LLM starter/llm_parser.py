"""
LLM Semantic Parser — Step 1 of the Generation Pipeline
========================================================
将自然语言场景描述转换为结构化 JSON，供 Transformer 时空编码使用。

输入示例:
  "阴天下午，一架四旋翼无人机在城市高楼背景中从远处50米飞近到200米，
   从下方仰拍，需要RGB模态"

输出: 结构化 JSON（见 SceneSpec schema）
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Literal

# ============================================================
# 1. 枚举词典
# ============================================================

class DroneType(str, Enum):
    QUADROTOR = "quadrotor"
    HEXACOPTER = "hexacopter"
    FIXED_WING = "fixed_wing"
    # 预留扩展
    # VTOL = "vtol"
    # OCTOCOPTER = "octocopter"


class DroneAction(str, Enum):
    HOVER = "hover"             # 悬停
    APPROACH = "approach"       # 飞近
    RETREAT = "retreat"         # 飞远
    LATERAL_MOVE = "lateral_move"  # 横向移动
    ASCEND = "ascend"               # 上升
    DESCEND = "descend"             # 下降
    CIRCLE = "circle"               # 盘旋
    NOISE = "noise"                 # 无规律微动


class Weather(str, Enum):
    CLEAR = "clear"             # 晴天
    OVERCAST = "overcast"       # 阴天
    RAINY = "rainy"             # 雨天
    FOGGY = "foggy"             # 雾天
    DUSTY = "dusty"             # 沙尘
    BACKLIGHT = "backlight"     # 逆光


class TimeOfDay(str, Enum):
    DAWN = "dawn"               # 黎明
    MORNING = "morning"         # 上午
    AFTERNOON = "afternoon"     # 下午
    DUSK = "dusk"               # 黄昏
    NIGHT = "night"             # 夜间


class Modality(str, Enum):
    RGB = "RGB"
    IR = "IR"


class CameraPosition(str, Enum):
    """
    相机相对于无人机的位置。

    语义（以无人机为中心）：
      - bottom: 相机在下 → 仰拍 → 看到无人机腹部
      - top:    相机在上 → 俯拍 → 看到无人机顶部
      - front:  相机在前 → 正前方视角
      - side:   相机在侧 → 侧面视角
      - back:   相机在后 → 尾部视角
    """
    BOTTOM = "bottom"   # 仰拍
    TOP = "top"         # 俯拍
    FRONT = "front"     # 前方
    SIDE = "side"       # 侧面
    BACK = "back"       # 后方


class SceneType(str, Enum):
    URBAN = "urban"             # 城市
    RURAL = "rural"             # 乡村
    MOUNTAIN = "mountain"       # 山地
    COASTAL = "coastal"         # 沿海
    DESERT = "desert"           # 沙漠
    FOREST = "forest"           # 森林
    INDUSTRIAL = "industrial"   # 工业区
    AIRFIELD = "airfield"       # 机场/机场周边


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
    "盘旋": DroneAction.CIRCLE, "绕圈": DroneAction.CIRCLE, "环绕": DroneAction.CIRCLE,
    "circle": DroneAction.CIRCLE,
    "微动": DroneAction.NOISE, "抖动": DroneAction.NOISE, "noise": DroneAction.NOISE,
}

WEATHER_KEYWORDS = {
    "晴": Weather.CLEAR, "晴天": Weather.CLEAR, "晴朗": Weather.CLEAR,
    "clear": Weather.CLEAR, "sunny": Weather.CLEAR,
    "阴": Weather.OVERCAST, "阴天": Weather.OVERCAST, "多云": Weather.OVERCAST,
    "overcast": Weather.OVERCAST, "cloudy": Weather.OVERCAST,
    "雨": Weather.RAINY, "雨天": Weather.RAINY, "下雨": Weather.RAINY,
    "rain": Weather.RAINY, "rainy": Weather.RAINY,
    "雾": Weather.FOGGY, "雾天": Weather.FOGGY, "雾气": Weather.FOGGY,
    "fog": Weather.FOGGY, "foggy": Weather.FOGGY,
    "沙尘": Weather.DUSTY, "沙": Weather.DUSTY, "尘": Weather.DUSTY,
    "dust": Weather.DUSTY, "sand": Weather.DUSTY,
    "逆光": Weather.BACKLIGHT, "背光": Weather.BACKLIGHT,
    "backlight": Weather.BACKLIGHT,
}

TIME_KEYWORDS = {
    "黎明": TimeOfDay.DAWN, "破晓": TimeOfDay.DAWN, "dawn": TimeOfDay.DAWN,
    "上午": TimeOfDay.MORNING, "早晨": TimeOfDay.MORNING, "早上": TimeOfDay.MORNING,
    "morning": TimeOfDay.MORNING,
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
    "城市": SceneType.URBAN, "都市": SceneType.URBAN, "高楼": SceneType.URBAN,
    "街道": SceneType.URBAN, "urban": SceneType.URBAN, "city": SceneType.URBAN,
    "乡村": SceneType.RURAL, "农村": SceneType.RURAL, "田野": SceneType.RURAL,
    "rural": SceneType.RURAL, "countryside": SceneType.RURAL,
    "山地": SceneType.MOUNTAIN, "山": SceneType.MOUNTAIN, "山脉": SceneType.MOUNTAIN,
    "mountain": SceneType.MOUNTAIN,
    "沿海": SceneType.COASTAL, "海岸": SceneType.COASTAL, "海": SceneType.COASTAL,
    "coastal": SceneType.COASTAL, "ocean": SceneType.COASTAL,
    "沙漠": SceneType.DESERT, "荒漠": SceneType.DESERT, "desert": SceneType.DESERT,
    "森林": SceneType.FOREST, "树林": SceneType.FOREST, "森林": SceneType.FOREST,
    "forest": SceneType.FOREST, "woods": SceneType.FOREST,
    "工业": SceneType.INDUSTRIAL, "工厂": SceneType.INDUSTRIAL,
    "industrial": SceneType.INDUSTRIAL, "factory": SceneType.INDUSTRIAL,
    "机场": SceneType.AIRFIELD, "跑道": SceneType.AIRFIELD,
    "airfield": SceneType.AIRFIELD, "airport": SceneType.AIRFIELD,
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
    # 可选：画面中的归一化位置 (0-1)
    norm_u: float = 0.5       # 归一化水平位置
    norm_v: float = 0.5       # 归一化垂直位置


@dataclass
class CameraSpec:
    """相机参数（相对于无人机）"""
    position: CameraPosition  # 相机相对无人机的位置
    elevation_deg: float = 30.0   # 仰角（度，bottom时典型值30-60）
    fov_deg: float = 60.0         # 视场角


@dataclass
class SceneSpec:
    """Step 1 输出：完整场景描述 JSON"""
    # 无人机属性
    drone_type: DroneType
    trajectory: list[TrajectoryPoint]   # 逐时间步轨迹

    # 环境属性
    time_of_day: TimeOfDay
    weather: Weather
    scene_type: SceneType             # ⭐ 新增：背景场景类型
    scene_description: str = ""       # ⭐ 新增：场景自然语言描述（用于日志/调试）

    # 模态与相机
    modality: Modality = Modality.RGB
    camera: CameraSpec = field(default_factory=lambda: CameraSpec(position=CameraPosition.FRONT))

    # 元数据
    raw_input: str = ""               # 原始用户输入
    confidence_note: str = ""         # LLM 附注（低置信度推断等）


# ============================================================
# 3. Few-shot Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个无人机场景解析器。将用户对无人机侦察场景的自然语言描述精确转换为 JSON。

## 字段说明

| 字段 | 类型 | 可选值 | 说明 |
|:--|:--|:--|:--|
| drone_type | string | quadrotor | 无人机类型（目前仅支持四旋翼） |
| trajectory | array | 见示例 | 逐时间步的轨迹：t(时间步序号), action, distance(米), norm_u(0-1水平位置), norm_v(0-1垂直位置) |
| time_of_day | string | dawn/morning/afternoon/dusk/night | 时段 |
| weather | string | clear/overcast/rainy/foggy/dusty/backlight | 天气 |
| scene_type | string | urban/rural/mountain/coastal/desert/forest/industrial/airfield | 背景场景类型 |
| scene_description | string | 自由文本 | 背景场景的简短自然语言描述 |
| modality | string | RGB/IR | 成像模态 |
| camera | object | position(相机相对无人机的位置: bottom仰拍/top俯拍/front前方/side侧面/back后方), elevation_deg(仰角度数), fov_deg(视场角) | 相机参数 |

## 相机位置语义

以无人机为参照中心：
- **bottom**: 相机在无人机下方 → 仰拍 → 看见无人机腹部/底部
- **top**: 相机在无人机上方 → 俯拍 → 看见无人机顶部
- **front**: 相机在无人机前方 → 正面视角
- **side**: 相机在无人机侧面 → 侧面视角
- **back**: 相机在无人机后方 → 尾部视角

## 规则

1. 轨迹至少2个时间步（起点+终点），如果用户描述了多个阶段则拆分为多个时间步
2. distance 是无人机距相机的距离（米），无人机越远→画面占比越小
3. 如果用户未指定某个字段，根据上下文合理推断，在 confidence_note 中标注
4. norm_u/norm_v 是无人机在画面中的归一化位置：默认(0.5, 0.5)即画面中央
5. 输出只包含 JSON，不要有其他文字，不要 markdown 代码块标记
6. scene_description 用英文描述，保持与 ControlNet prompt 风格一致

## 示例

输入: "晴天下午，城市高楼背景，四旋翼从50米外正面飞来，接近到200米处，RGB"
输出:
{"drone_type":"quadrotor","trajectory":[{"t":0,"action":"approach","distance":50.0,"norm_u":0.5,"norm_v":0.5},{"t":1,"action":"approach","distance":200.0,"norm_u":0.5,"norm_v":0.5}],"time_of_day":"afternoon","weather":"clear","scene_type":"urban","scene_description":"city high-rise buildings with clear sky","modality":"RGB","camera":{"position":"front","elevation_deg":15.0,"fov_deg":60.0},"confidence_note":""}

输入: "雾天黄昏，一架无人机在森林上空盘旋，从下方仰拍，距离约100米"
输出:
{"drone_type":"quadrotor","trajectory":[{"t":0,"action":"hover","distance":100.0,"norm_u":0.5,"norm_v":0.5}],"time_of_day":"dusk","weather":"foggy","scene_type":"forest","scene_description":"dense forest canopy with misty atmosphere at dusk","modality":"IR","camera":{"position":"bottom","elevation_deg":45.0,"fov_deg":60.0},"confidence_note":"未指定modality，根据雾天低能见度场景推断为IR"}

输入: "大晴天上午，沙漠地带，四旋翼从100米远处横向移动到右侧，侧面拍摄"
输出:
{"drone_type":"quadrotor","trajectory":[{"t":0,"action":"lateral_move","distance":100.0,"norm_u":0.3,"norm_v":0.5},{"t":1,"action":"lateral_move","distance":100.0,"norm_u":0.7,"norm_v":0.5}],"time_of_day":"morning","weather":"clear","scene_type":"desert","scene_description":"vast desert landscape under bright morning sun","modality":"RGB","camera":{"position":"side","elevation_deg":20.0,"fov_deg":60.0},"confidence_note":""}

输入: "阴天早晨，在工业区场景中，无人机从200米远处飞近到50米，从后方拍摄尾部视角"
输出:
{"drone_type":"quadrotor","trajectory":[{"t":0,"action":"approach","distance":200.0,"norm_u":0.5,"norm_v":0.5},{"t":1,"action":"approach","distance":50.0,"norm_u":0.5,"norm_v":0.5}],"time_of_day":"morning","weather":"overcast","scene_type":"industrial","scene_description":"industrial complex with factories and smokestacks under overcast sky","modality":"RGB","camera":{"position":"back","elevation_deg":10.0,"fov_deg":60.0},"confidence_note":""}

输入: "雨天的黎明的海岸边，无人机在高空盘旋，从顶部俯拍，距离150米，IR模态"
输出:
{"drone_type":"quadrotor","trajectory":[{"t":0,"action":"hover","distance":150.0,"norm_u":0.5,"norm_v":0.3}],"time_of_day":"dawn","weather":"rainy","scene_type":"coastal","scene_description":"rocky coastline with rough seas under rain at dawn","modality":"IR","camera":{"position":"top","elevation_deg":75.0,"fov_deg":60.0},"confidence_note":"俯拍时无人机在画面偏上方(norm_v=0.3)"}
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
        action_str = pt["action"]
        action = _coerce_enum(action_str, DroneAction, ACTION_KEYWORDS, "hover")
        traj_list.append(TrajectoryPoint(
            t=pt.get("t", 0),
            action=action,
            distance=float(pt.get("distance", 100)),
            norm_u=float(pt.get("norm_u", 0.5)),
            norm_v=float(pt.get("norm_v", 0.5)),
        ))

    # 解析 camera
    cam_data = data.get("camera", {})
    cam_pos = _coerce_enum(
        cam_data.get("position", "front"), CameraPosition, CAMERA_KEYWORDS, "front"
    )

    spec = SceneSpec(
        drone_type=_coerce_enum(data.get("drone_type", "quadrotor"), DroneType, {}, "quadrotor"),
        trajectory=traj_list,
        time_of_day=_coerce_enum(data.get("time_of_day", "afternoon"), TimeOfDay, TIME_KEYWORDS, "afternoon"),
        weather=_coerce_enum(data.get("weather", "clear"), Weather, WEATHER_KEYWORDS, "clear"),
        scene_type=_coerce_enum(data.get("scene_type", "urban"), SceneType, SCENE_KEYWORDS, "urban"),
        scene_description=data.get("scene_description", ""),
        modality=_coerce_enum(data.get("modality", "RGB"), Modality, {"ir": Modality.IR, "rgb": Modality.RGB, "热红外": Modality.IR, "红外": Modality.IR}, "RGB"),
        camera=CameraSpec(
            position=cam_pos,
            elevation_deg=float(cam_data.get("elevation_deg", 30)),
            fov_deg=float(cam_data.get("fov_deg", 60)),
        ),
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
# 6. 便捷函数：一步解析
# ============================================================

def parse(user_input: str, backend: Literal["openai", "claude", "deepseek", "dry"] = "dry",
          model: str = "", api_key: str = "") -> SceneSpec:
    """
    一步解析用户输入。

    Args:
        user_input: 自然语言场景描述
        backend: "openai" | "claude" | "dry"（dry-run 返回默认值用于测试）
        model: 模型名（空则用默认值）
        api_key: API key（空则从环境变量取）

    Returns:
        SceneSpec: 结构化场景参数
    """
    if backend == "dry":
        # dry-run: 不调 LLM，返回带有 raw_input 的默认 spec
        return SceneSpec(
            drone_type=DroneType.QUADROTOR,
            trajectory=[TrajectoryPoint(t=0, action=DroneAction.HOVER, distance=100.0)],
            time_of_day=TimeOfDay.AFTERNOON,
            weather=Weather.CLEAR,
            scene_type=SceneType.URBAN,
            raw_input=user_input,
            confidence_note="[dry-run] 未调用 LLM",
        )
    elif backend == "openai":
        return parse_with_openai(user_input, model=model or "gpt-4o", api_key=api_key)
    elif backend == "claude":
        return parse_with_claude(user_input, model=model or "claude-sonnet-4-20250514", api_key=api_key)
    elif backend == "deepseek":
        return parse_with_deepseek(user_input, model=model or "deepseek-chat", api_key=api_key)
    else:
        raise ValueError(f"不支持的 backend: {backend}")


# ============================================================
# 7. 工具函数
# ============================================================

def scene_spec_to_dict(spec: SceneSpec) -> dict:
    """将 SceneSpec 转为普通 dict（轨迹中的枚举值转为字符串）。"""
    d = asdict(spec)
    for pt in d["trajectory"]:
        pt["action"] = pt["action"].value if hasattr(pt["action"], "value") else str(pt["action"])
    d["drone_type"] = spec.drone_type.value
    d["time_of_day"] = spec.time_of_day.value
    d["weather"] = spec.weather.value
    d["scene_type"] = spec.scene_type.value
    d["modality"] = spec.modality.value
    d["camera"]["position"] = spec.camera.position.value
    return d


def scene_spec_to_json(spec: SceneSpec, indent: int = 2) -> str:
    """SceneSpec → 紧凑 JSON 字符串。"""
    return json.dumps(scene_spec_to_dict(spec), ensure_ascii=False, indent=indent)


# ============================================================
# 8. CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM Parser: 自然语言 → 场景 JSON")
    p.add_argument("input", nargs="?", help="场景描述文本，不提供则进入交互模式")
    p.add_argument("--backend", choices=["openai", "claude", "deepseek", "dry"], default="dry",
                   help="LLM 后端 (default: dry)")
    p.add_argument("--model", default="", help="覆盖默认模型")
    p.add_argument("--api-key", default="", help="API key（或设环境变量）")
    p.add_argument("--json", action="store_true", help="直接输出 JSON（不打印 Python repr）")
    args = p.parse_args()

    if args.input:
        spec = parse(args.input, backend=args.backend, model=args.model, api_key=args.api_key)
        if args.json:
            print(scene_spec_to_json(spec))
        else:
            print(scene_spec_to_json(spec))
    else:
        # 交互模式
        print("LLM Parser 交互模式 (Ctrl+C 退出)")
        print(f"后端: {args.backend}")
        print("-" * 50)
        try:
            while True:
                user = input("\n场景描述 > ").strip()
                if not user:
                    continue
                spec = parse(user, backend=args.backend, model=args.model, api_key=args.api_key)
                print(scene_spec_to_json(spec))
        except KeyboardInterrupt:
            print("\n退出。")
