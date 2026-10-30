"""
S1 JSON Validator 单元测试
==========================
覆盖 6 项检查的正常和异常路径。
"""

import sys
import os
import pytest

# 确保能 import s1_json_validator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s1_json_validator import (
    S1JSONValidator,
    A1_SCHEMA_INCOMPLETE,
    A1_ENUM_INVALID,
    A1_LOGIC_CONFLICT,
    A1_DESC_MISMATCH,
    A1_TRAJ_INVALID,
    A1_CAMERA_INVALID,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def validator():
    return S1JSONValidator(config_dir=CONFIG_DIR)


@pytest.fixture
def valid_data():
    """一个完全合法的 JSON 样本"""
    return {
        "drone_type": "quadcopter",
        "trajectory": [
            {"action": "fly_straight", "distance": 50.0, "t": 0.0},
            {"action": "fly_straight", "distance": 50.0, "t": 1.0},
            {"action": "hover", "distance": 0.0, "t": 2.0},
        ],
        "time_of_day": "afternoon",
        "weather": "clear",
        "scene_type": "outdoor_urban",
        "scene_description": "A quadcopter flying over a city park on a clear afternoon.",
        "modality": "RGB_IR_pair",
        "camera": {
            "position": "high_angle",
            "fov_deg": 60,
            "elevation_deg": 45,
        },
        "confidence_note": "standard generation",
    }


# ═══════════════════════════════════════════════════════
# 正常路径
# ═══════════════════════════════════════════════════════

class TestPass:
    def test_valid_full(self, validator, valid_data):
        r = validator.validate(valid_data)
        assert r["pass"] is True
        assert r["failure_code"] is None

    def test_trajectory_distance_zero_allowed(self, validator, valid_data):
        """悬停 distance=0 是合法的"""
        valid_data["trajectory"] = [{"action": "hover", "distance": 0.0, "t": 0.0}]
        r = validator.validate(valid_data)
        assert r["pass"] is True

    def test_camera_bottom_negative_elevation(self, validator, valid_data):
        """底部视角 + 负仰角 是合法的"""
        valid_data["camera"] = {"position": "bottom", "fov_deg": 80, "elevation_deg": -75}
        r = validator.validate(valid_data)
        assert r["pass"] is True

    def test_noon_fog_warns_but_passes(self, validator, valid_data):
        """正午+浓雾是 WARN，不截停"""
        valid_data["time_of_day"] = "noon"
        valid_data["weather"] = "foggy"
        r = validator.validate(valid_data)
        assert r["pass"] is True  # WARN 不截停


# ═══════════════════════════════════════════════════════
# 检查 1: Schema 完整性
# ═══════════════════════════════════════════════════════

class TestSchema:
    def test_missing_field(self, validator, valid_data):
        del valid_data["drone_type"]
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_SCHEMA_INCOMPLETE
        assert "drone_type" in r["reason"]

    def test_empty_trajectory(self, validator, valid_data):
        valid_data["trajectory"] = []
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_SCHEMA_INCOMPLETE

    def test_trajectory_not_list(self, validator, valid_data):
        valid_data["trajectory"] = "not_a_list"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_SCHEMA_INCOMPLETE

    def test_missing_camera_subfield(self, validator, valid_data):
        valid_data["camera"] = {"position": "high_angle"}  # 缺 fov_deg, elevation_deg
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_SCHEMA_INCOMPLETE


# ═══════════════════════════════════════════════════════
# 检查 2: 枚举值合法性
# ═══════════════════════════════════════════════════════

class TestEnums:
    def test_invalid_drone_type(self, validator, valid_data):
        valid_data["drone_type"] = "helicopter"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_ENUM_INVALID

    def test_invalid_weather(self, validator, valid_data):
        valid_data["weather"] = "tornado"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_ENUM_INVALID

    def test_invalid_time_of_day(self, validator, valid_data):
        valid_data["time_of_day"] = "midnight"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_ENUM_INVALID

    def test_invalid_camera_position(self, validator, valid_data):
        valid_data["camera"]["position"] = "underground"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_ENUM_INVALID


# ═══════════════════════════════════════════════════════
# 检查 3: 场景逻辑自洽
# ═══════════════════════════════════════════════════════

class TestLogic:
    def test_night_backlight_reject(self, validator, valid_data):
        valid_data["time_of_day"] = "night"
        valid_data["weather"] = "backlight"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_LOGIC_CONFLICT

    def test_indoor_rainy_reject(self, validator, valid_data):
        valid_data["scene_type"] = "indoor"
        valid_data["weather"] = "rainy"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_LOGIC_CONFLICT

    def test_indoor_snowy_reject(self, validator, valid_data):
        valid_data["scene_type"] = "indoor"
        valid_data["weather"] = "snowy"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_LOGIC_CONFLICT

    def test_bottom_elevation_positive_reject(self, validator, valid_data):
        valid_data["camera"] = {"position": "bottom", "fov_deg": 60, "elevation_deg": 10}
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_LOGIC_CONFLICT

    def test_no_conflict_passes(self, validator, valid_data):
        """没有触发任何互斥规则应通过"""
        valid_data["time_of_day"] = "morning"
        valid_data["weather"] = "cloudy"
        r = validator.validate(valid_data)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 4: scene_description 关键词交叉检查
# ═══════════════════════════════════════════════════════

class TestDescription:
    def test_sunny_in_overcast(self, validator, valid_data):
        valid_data["scene_description"] = "Bright sunny day with clear sky"
        valid_data["weather"] = "overcast"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_DESC_MISMATCH

    def test_rain_keyword_in_clear(self, validator, valid_data):
        valid_data["scene_description"] = "Heavy rain drizzle in the city"
        valid_data["weather"] = "clear"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_DESC_MISMATCH

    def test_night_keyword_in_morning(self, validator, valid_data):
        valid_data["scene_description"] = "Dark midnight scene"
        valid_data["time_of_day"] = "morning"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_DESC_MISMATCH

    def test_indoor_keyword_in_outdoor(self, validator, valid_data):
        valid_data["scene_description"] = "Drone flying inside a warehouse corridor"
        valid_data["scene_type"] = "outdoor_urban"
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_DESC_MISMATCH

    def test_no_conflict_passes(self, validator, valid_data):
        valid_data["scene_description"] = "Drone flying over a city on a clear day"
        valid_data["weather"] = "clear"
        r = validator.validate(valid_data)
        assert r["pass"] is True

    def test_no_description_passes(self, validator, valid_data):
        valid_data["scene_description"] = ""
        r = validator.validate(valid_data)
        assert r["pass"] is True  # 空描述不做交叉检查


# ═══════════════════════════════════════════════════════
# 检查 5: trajectory 值域
# ═══════════════════════════════════════════════════════

class TestTrajectory:
    def test_negative_distance(self, validator, valid_data):
        valid_data["trajectory"] = [{"action": "fly_straight", "distance": -5.0, "t": 0.0}]
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_TRAJ_INVALID

    def test_invalid_action(self, validator, valid_data):
        valid_data["trajectory"] = [{"action": "do_a_barrel_roll", "distance": 10.0, "t": 0.0}]
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_TRAJ_INVALID

    def test_t_not_increasing(self, validator, valid_data):
        valid_data["trajectory"] = [
            {"action": "fly_straight", "distance": 10.0, "t": 1.0},
            {"action": "fly_straight", "distance": 10.0, "t": 0.5},  # 倒退
        ]
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_TRAJ_INVALID

    def test_no_t_field_passes(self, validator, valid_data):
        """没有 t 字段就跳过 t 递增检查"""
        valid_data["trajectory"] = [
            {"action": "fly_straight", "distance": 10.0},
            {"action": "hover", "distance": 0.0},
        ]
        r = validator.validate(valid_data)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 6: camera 参数合理区间
# ═══════════════════════════════════════════════════════

class TestCamera:
    def test_fov_too_small(self, validator, valid_data):
        valid_data["camera"]["fov_deg"] = 10
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_CAMERA_INVALID

    def test_fov_too_large(self, validator, valid_data):
        valid_data["camera"]["fov_deg"] = 150
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_CAMERA_INVALID

    def test_elevation_out_of_range(self, validator, valid_data):
        """high_angle 的 elevation 应该在 [30, 60]"""
        valid_data["camera"]["position"] = "high_angle"
        valid_data["camera"]["elevation_deg"] = 80  # 超出
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_CAMERA_INVALID

    def test_top_down_low_elevation(self, validator, valid_data):
        """top_down 的 elevation 应该在 [60, 90]"""
        valid_data["camera"]["position"] = "top_down"
        valid_data["camera"]["elevation_deg"] = 30  # 太低了
        r = validator.validate(valid_data)
        assert r["pass"] is False
        assert r["failure_code"] == A1_CAMERA_INVALID


# ═══════════════════════════════════════════════════════
# 短路求值验证
# ═══════════════════════════════════════════════════════

class TestShortCircuit:
    def test_schema_fails_first(self, validator, valid_data):
        """schema 失败应直接返回，不检查后续"""
        valid_data["drone_type"] = "INVALID"  # 先触发枚举失败而非 schema
        valid_data["weather"] = "INVALID"
        r = validator.validate(valid_data)
        # 枚举检查在 schema 之后 → 如果 schema 通过则第一个失败是枚举
        assert r["failure_code"] == A1_ENUM_INVALID

    def test_enum_before_logic(self, validator, valid_data):
        """枚举失败应优先于逻辑矛盾"""
        valid_data["drone_type"] = "INVALID"
        valid_data["time_of_day"] = "night"
        valid_data["weather"] = "backlight"  # 逻辑矛盾但不会到达
        r = validator.validate(valid_data)
        assert r["failure_code"] == A1_ENUM_INVALID  # 不是 A1_LOGIC_CONFLICT
