"""
S2 Transformer Validator 单元测试
==================================
覆盖 6 项检查的正常和异常路径。
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s2_transformer_validator import (
    S2TransformerValidator,
    A2_BBOX_OOB,
    A2_SIZE_ANOMALY,
    A2_POSITION_JUMP,
    A2_SEG_EMPTY,
    A2_DEPTH_FLAT,
    A2_FRAME_COUNT_MISMATCH,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def validator():
    return S2TransformerValidator(config_dir=CONFIG_DIR, frame_size=(512, 512))


def _frames(*bboxes):
    """快捷构造 frames 列表"""
    return [{"bbox": b, "t": float(i)} for i, b in enumerate(bboxes)]


# ═══════════════════════════════════════════════════════
# 正常路径
# ═══════════════════════════════════════════════════════

class TestPass:
    def test_valid_simple(self, validator):
        frames = _frames((100, 200, 30, 30), (102, 198, 32, 32), (105, 195, 34, 34))
        r = validator.validate(frames, trajectory_len=3, drone_type="quadcopter")
        assert r["pass"] is True

    def test_single_frame(self, validator):
        """单帧不检查帧间关系"""
        frames = _frames((100, 200, 30, 30))
        r = validator.validate(frames, trajectory_len=1, drone_type="quadcopter")
        assert r["pass"] is True

    def test_bbox_on_edge(self, validator):
        """bbox 刚好在边界上应通过"""
        frames = _frames((482, 482, 30, 30))  # 右下角刚好贴边
        r = validator.validate(frames, trajectory_len=1, drone_type="quadcopter")
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 1: bbox 越界
# ═══════════════════════════════════════════════════════

class TestBboxBounds:
    def test_negative_x(self, validator):
        frames = _frames((-5, 100, 30, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB

    def test_negative_y(self, validator):
        frames = _frames((100, -10, 30, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB

    def test_overflow_right(self, validator):
        frames = _frames((500, 100, 30, 30))  # x+w=530 > 512
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB

    def test_overflow_bottom(self, validator):
        frames = _frames((100, 500, 30, 30))  # y+h=530 > 512
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB

    def test_zero_size(self, validator):
        frames = _frames((100, 100, 0, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB

    def test_negative_size(self, validator):
        frames = _frames((100, 100, -5, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_BBOX_OOB


# ═══════════════════════════════════════════════════════
# 检查 2: 帧间尺寸渐变
# ═══════════════════════════════════════════════════════

class TestSizeSmoothness:
    def test_normal_growth(self, validator):
        """缓慢增长应通过"""
        frames = _frames((100, 200, 30, 30), (100, 200, 35, 35), (100, 200, 40, 40))
        r = validator.validate(frames, trajectory_len=3)
        assert r["pass"] is True

    def test_sudden_jump(self, validator):
        """宽度突变 30%+ 应失败（40→60 = 50% 突变）"""
        frames = _frames((100, 200, 40, 30), (100, 200, 60, 30))
        r = validator.validate(frames, trajectory_len=2)
        assert r["pass"] is False
        assert r["failure_code"] == A2_SIZE_ANOMALY

    def test_shrink_within_threshold(self, validator):
        """缩小在阈值内应通过"""
        frames = _frames((100, 200, 40, 30), (100, 200, 35, 30))
        r = validator.validate(frames, trajectory_len=2)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 3: 帧间位置连续
# ═══════════════════════════════════════════════════════

class TestPositionContinuity:
    def test_small_move(self, validator):
        """小幅移动应通过"""
        frames = _frames((100, 200, 30, 30), (110, 210, 30, 30))
        r = validator.validate(frames, trajectory_len=2, drone_type="quadcopter")
        assert r["pass"] is True

    def test_large_jump(self, validator):
        """大幅跳跃（200px）应失败"""
        frames = _frames((100, 100, 30, 30), (300, 300, 30, 30))
        r = validator.validate(frames, trajectory_len=2, drone_type="quadcopter")
        assert r["pass"] is False
        assert r["failure_code"] == A2_POSITION_JUMP

    def test_fpv_allows_faster(self, validator):
        """FPV 的 v_max=50，小幅跳跃应通过"""
        frames = _frames((100, 100, 30, 30), (140, 130, 30, 30))  # ~50px
        r = validator.validate(frames, trajectory_len=2, drone_type="fpv_drone")
        assert r["pass"] is True

    def test_fpv_rejects_too_fast(self, validator):
        """FPV 大幅跳跃仍应失败"""
        frames = _frames((100, 100, 30, 30), (300, 300, 30, 30))
        r = validator.validate(frames, trajectory_len=2, drone_type="fpv_drone")
        assert r["pass"] is False


# ═══════════════════════════════════════════════════════
# 检查 4: seg 图非空
# ═══════════════════════════════════════════════════════

class TestSegNonempty:
    def test_seg_with_content(self, validator):
        frames = _frames((100, 200, 30, 30))
        frames[0]["seg_map"] = np.ones((512, 512), dtype=np.uint8) * 255
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is True

    def test_seg_empty(self, validator):
        frames = _frames((100, 200, 30, 30))
        frames[0]["seg_map"] = np.zeros((512, 512), dtype=np.uint8)
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_SEG_EMPTY

    def test_no_seg_ok(self, validator):
        """seg 缺失时不检查"""
        frames = _frames((100, 200, 30, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 5: depth 图有效性
# ═══════════════════════════════════════════════════════

class TestDepthValidity:
    def test_valid_depth(self, validator):
        frames = _frames((100, 200, 30, 30))
        frames[0]["depth_map"] = np.random.rand(512, 512).astype(np.float32)
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is True

    def test_all_zero_depth(self, validator):
        frames = _frames((100, 200, 30, 30))
        frames[0]["depth_map"] = np.zeros((512, 512), dtype=np.float32)
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_DEPTH_FLAT

    def test_near_flat_depth(self, validator):
        """方差极小的 depth 应告警"""
        frames = _frames((100, 200, 30, 30))
        frames[0]["depth_map"] = np.full((512, 512), 0.5, dtype=np.float32) + \
                                  np.random.randn(512, 512).astype(np.float32) * 0.001
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is False
        assert r["failure_code"] == A2_DEPTH_FLAT

    def test_no_depth_ok(self, validator):
        """depth 缺失时不检查"""
        frames = _frames((100, 200, 30, 30))
        r = validator.validate(frames, trajectory_len=1)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 6: 帧数 vs trajectory
# ═══════════════════════════════════════════════════════

class TestFrameCount:
    def test_match(self, validator):
        frames = _frames((100, 200, 30, 30), (110, 210, 30, 30))
        r = validator.validate(frames, trajectory_len=2)
        assert r["pass"] is True

    def test_mismatch(self, validator):
        frames = _frames((100, 200, 30, 30))
        r = validator.validate(frames, trajectory_len=3)
        assert r["pass"] is False
        assert r["failure_code"] == A2_FRAME_COUNT_MISMATCH

    def test_empty_frames_vs_trajectory(self, validator):
        r = validator.validate([], trajectory_len=5)
        assert r["pass"] is False
        assert r["failure_code"] == A2_FRAME_COUNT_MISMATCH


# ═══════════════════════════════════════════════════════
# 短路求值
# ═══════════════════════════════════════════════════════

class TestShortCircuit:
    def test_bbox_fails_first(self, validator):
        """bbox 越界应优先于尺寸检查"""
        frames = _frames((-5, 100, 30, 30), (100, 100, 60, 30))  # 既越界又突变
        r = validator.validate(frames, trajectory_len=2)
        assert r["failure_code"] == A2_BBOX_OOB  # 先检 1

    def test_size_before_position(self, validator):
        """尺寸异常应先于位置检查"""
        frames = _frames((100, 100, 40, 30), (200, 200, 60, 30))  # 尺寸50%突变 + 位置大幅跳跃
        r = validator.validate(frames, trajectory_len=2)
        assert r["failure_code"] == A2_SIZE_ANOMALY  # 先检 2 而非检 3
